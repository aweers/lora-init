"""
Test different LoRA initialization strategies on a small LLM using Unsloth.

Compares:
1. Standard init: A randomly initialized (Kaiming), B = 0
2. Reversed init: A = 0, B randomly initialized (Kaiming)
3. Orthogonal QR init: A and B initialized using QR decomposition orthogonal matrix
4. Orthogonal Eye init: A and B initialized using identity matrix orthogonal basis

Logs metrics to Weights & Biases.
"""

from unsloth import FastLanguageModel
import argparse
import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from peft.tuners.lora import LoraLayer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

import wandb

EVAL_STEPS = 40

MODEL_CONFIG = {
    "hf_name": "Qwen/Qwen2.5-1.5B-Instruct",
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
}

DATASET_CONFIG = {
    "name": "HuggingFaceH4/no_robots",
}


@dataclass
class RunMetrics:
    """Metrics collected during a single training run."""

    seed: int
    train_losses: list = field(default_factory=list)
    eval_losses: list = field(default_factory=list)
    perplexities: list = field(default_factory=list)
    best_eval_loss: float = float("inf")
    best_perplexity: float = float("inf")
    final_train_loss: float = float("inf")


def compute_aggregate_stats(metrics_list: list[RunMetrics]) -> dict:
    """Compute mean and std statistics across multiple runs."""
    min_train_steps = min(len(m.train_losses) for m in metrics_list)
    min_eval_epochs = min(len(m.eval_losses) for m in metrics_list)

    train_losses_array = np.array(
        [m.train_losses[:min_train_steps] for m in metrics_list]
    )
    train_loss_mean = train_losses_array.mean(axis=0)
    train_loss_std = train_losses_array.std(axis=0)

    eval_losses_array = np.array(
        [m.eval_losses[:min_eval_epochs] for m in metrics_list]
    )
    eval_loss_mean = eval_losses_array.mean(axis=0)
    eval_loss_std = eval_losses_array.std(axis=0)

    perplexities_array = np.array(
        [m.perplexities[:min_eval_epochs] for m in metrics_list]
    )
    perplexity_mean = perplexities_array.mean(axis=0)
    perplexity_std = perplexities_array.std(axis=0)

    best_eval_losses = np.array([m.best_eval_loss for m in metrics_list])
    best_perplexities = np.array([m.best_perplexity for m in metrics_list])
    final_train_losses = np.array([m.final_train_loss for m in metrics_list])

    return {
        "train_loss_mean": train_loss_mean,
        "train_loss_std": train_loss_std,
        "eval_loss_mean": eval_loss_mean,
        "eval_loss_std": eval_loss_std,
        "perplexity_mean": perplexity_mean,
        "perplexity_std": perplexity_std,
        "best_eval_loss_mean": best_eval_losses.mean(),
        "best_eval_loss_std": best_eval_losses.std(),
        "best_perplexity_mean": best_perplexities.mean(),
        "best_perplexity_std": best_perplexities.std(),
        "final_train_loss_mean": final_train_losses.mean(),
        "final_train_loss_std": final_train_losses.std(),
    }


def get_args():
    parser = argparse.ArgumentParser(
        description="Test LoRA initialization strategies with Unsloth",
    )
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--grad_accum_steps", type=int, default=4, help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-4, help="Learning rate"
    )
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument(
        "--max_length", type=int, default=1024, help="Max sequence length"
    )
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument(
        "--num_runs",
        type=int,
        default=10,
        help="Number of runs per experiment with different seeds",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="lora-init-comparison-unsloth",
        help="W&B project name",
    )
    parser.add_argument(
        "--init_strategy",
        type=str,
        default="both",
        choices=["standard", "reversed", "orthogonal_qr", "orthogonal_eye", "both"],
        help="Which initialization strategy to test",
    )
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def reinitialize_lora_weights(
    model: nn.Module,
    init_type: Literal["standard", "reversed", "orthogonal_qr", "orthogonal_eye"],
    lora_alpha: int,
    lora_r: int,
):
    """
    Reinitialize LoRA weights with specified strategy.
    Why use `a=math.sqrt(5)`? See
    https://github.com/pytorch/pytorch/blob/d38164a545b4a4e4e0cf73ce67173f70574890b6/torch/nn/modules/linear.py#L117
    and https://github.com/pytorch/pytorch/issues/57109

    Args:
        model: The PEFT model
        init_type: Initialization strategy:
            - 'standard': A=random (Kaiming), B=0
            - 'reversed': A=0, B=random (Kaiming)
            - 'orthogonal_qr': Orthogonal init using QR decomposition
            - 'orthogonal_eye': Orthogonal init using identity matrix
        lora_alpha: LoRA alpha for scaling
        lora_r: LoRA rank
    """
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            for adapter_name in module.lora_A.keys():
                lora_A = module.lora_A[adapter_name]
                lora_B = module.lora_B[adapter_name]

                if init_type == "standard":
                    nn.init.kaiming_uniform_(lora_A.weight, a=math.sqrt(5))
                    nn.init.zeros_(lora_B.weight)
                elif init_type == "reversed":
                    nn.init.zeros_(lora_A.weight)
                    nn.init.kaiming_uniform_(lora_B.weight, a=math.sqrt(5))
                elif init_type == "orthogonal_qr":
                    r = lora_A.weight.shape[0]
                    in_features = lora_A.weight.shape[1]
                    out_features = lora_B.weight.shape[0]

                    with torch.no_grad():
                        # QR decomposition of a random matrix
                        X = torch.randn(r, r)
                        Q, _ = torch.linalg.qr(X)

                        # Split into two sets (alternating rows)
                        set1 = Q[0::2, :]  # Rows at indices 0, 2, 4, ...
                        set2 = Q[1::2, :]  # Rows at indices 1, 3, 5, ...

                        # Compute Kaiming-style scaling for variance matching
                        # For Kaiming uniform with a=sqrt(5), variance ≈ 1/fan_in
                        a_scale = math.sqrt(1.0 / in_features)
                        b_scale = math.sqrt(1.0 / r)

                        # Initialize A and B following the orthogonal pattern
                        a_wt = torch.randn(in_features, r//2).mm(set1).T * a_scale
                        b_wt = torch.randn(r//2, out_features).T.mm(set2) * b_scale

                        # Copy to weights with proper dtype
                        lora_A.weight.copy_(a_wt.to(lora_A.weight.dtype))
                        lora_B.weight.copy_(b_wt.to(lora_B.weight.dtype))
                elif init_type == "orthogonal_eye":
                    r = lora_A.weight.shape[0]
                    in_features = lora_A.weight.shape[1]
                    out_features = lora_B.weight.shape[0]

                    with torch.no_grad():
                        # Use identity matrix
                        Q = torch.eye(r, r)

                        # Split into two sets (alternating rows)
                        set1 = Q[0::2, :]  # Rows at indices 0, 2, 4, ...
                        set2 = Q[1::2, :]  # Rows at indices 1, 3, 5, ...

                        # Compute Kaiming-style scaling for variance matching
                        # For Kaiming uniform with a=sqrt(5), variance ≈ 1/fan_in
                        a_scale = math.sqrt(1.0 / in_features)
                        b_scale = math.sqrt(1.0 / r)

                        # Initialize A and B following the orthogonal pattern
                        a_wt = torch.randn(in_features, r//2).mm(set1).T * a_scale
                        b_wt = torch.randn(r//2, out_features).T.mm(set2) * b_scale

                        # Copy to weights with proper dtype
                        lora_A.weight.copy_(a_wt.to(lora_A.weight.dtype))
                        lora_B.weight.copy_(b_wt.to(lora_B.weight.dtype))

                module.scaling[adapter_name] = lora_alpha / lora_r

    return model


def format_no_robots(example):
    """Format no_robots dataset example into a single text string."""
    messages = example.get("messages", [])
    if isinstance(messages, list) and len(messages) > 0:
        formatted_parts = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                formatted_parts.append(f"{role.capitalize()}: {content}")
        return "\n\n".join(formatted_parts)
    return str(messages)


def prepare_dataset(tokenizer, max_length, split="train"):
    """Prepare and tokenize the no_robots dataset."""
    dataset = load_dataset(DATASET_CONFIG["name"], split=split)

    def group_texts(examples):
        texts = []
        for i in range(len(examples[list(examples.keys())[0]])):
            example = {k: v[i] for k, v in examples.items()}
            text = format_no_robots(example)
            if text and text.strip():
                texts.append(text.strip())

        if not texts:
            return {"input_ids": [], "attention_mask": [], "labels": []}

        tokenized = tokenizer(texts, truncation=False, padding=False)

        concatenated_ids = []
        for ids in tokenized["input_ids"]:
            concatenated_ids.extend(ids)
            concatenated_ids.append(tokenizer.eos_token_id)

        total_length = len(concatenated_ids)
        total_length = (total_length // max_length) * max_length

        result = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }

        for i in range(0, total_length, max_length):
            chunk = concatenated_ids[i : i + max_length]
            result["input_ids"].append(chunk)
            result["attention_mask"].append([1] * max_length)
            result["labels"].append(chunk)

        return result

    tokenized_dataset = dataset.map(
        group_texts,
        batched=True,
        remove_columns=dataset.column_names,
        desc=f"Tokenizing {split}",
    )

    tokenized_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

    return tokenized_dataset


def collate_fn_with_padding(batch, pad_token_id=0):
    """Pad variable-length sequences in a batch."""
    batch = [item for item in batch if len(item.get("input_ids", [])) > 0]

    if not batch:
        return {
            "input_ids": torch.tensor([], dtype=torch.long),
            "attention_mask": torch.tensor([], dtype=torch.long),
            "labels": torch.tensor([], dtype=torch.long),
        }

    max_length = max(len(item["input_ids"]) for item in batch)
    input_ids_padded = []
    attention_mask_padded = []
    labels_padded = []

    for item in batch:
        input_ids = item["input_ids"]
        attention_mask = item["attention_mask"]
        labels = item["labels"]
        padding_length = max_length - len(input_ids)

        if isinstance(input_ids, torch.Tensor):
            input_ids_padded.append(
                torch.nn.functional.pad(
                    input_ids, (0, padding_length), value=pad_token_id
                )
            )
        else:
            input_ids_padded.append(
                torch.tensor(
                    input_ids + [pad_token_id] * padding_length, dtype=torch.long
                )
            )

        if isinstance(attention_mask, torch.Tensor):
            attention_mask_padded.append(
                torch.nn.functional.pad(attention_mask, (0, padding_length), value=0)
            )
        else:
            attention_mask_padded.append(
                torch.tensor(attention_mask + [0] * padding_length, dtype=torch.long)
            )

        if isinstance(labels, torch.Tensor):
            labels_padded.append(
                torch.nn.functional.pad(labels, (0, padding_length), value=-100)
            )
        else:
            labels_padded.append(
                torch.tensor(labels + [-100] * padding_length, dtype=torch.long)
            )

    return {
        "input_ids": torch.stack(input_ids_padded),
        "attention_mask": torch.stack(attention_mask_padded),
        "labels": torch.stack(labels_padded),
    }


def compute_metrics(model, dataloader, device):
    """Compute evaluation metrics."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            num_tokens = (batch["labels"] != -100).sum().item()
            total_loss += outputs.loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float("inf")

    return {"eval_loss": avg_loss, "perplexity": perplexity}


def train_with_init(
    init_type: Literal["standard", "reversed"],
    args,
    tokenizer,
    train_dataset,
    eval_dataset,
    device,
    seed: int,
    run_number: int,
) -> RunMetrics:
    """Train model with specified LoRA initialization using Unsloth."""
    set_seed(seed)

    run_name = (
        f"lora-{init_type}-init-r{args.lora_r}-run{run_number}-seed{seed}-unsloth"
    )
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            "model_name": MODEL_CONFIG["hf_name"],
            "dataset": DATASET_CONFIG["name"],
            "init_type": init_type,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "effective_batch_size": args.batch_size * args.grad_accum_steps,
            "num_epochs": args.num_epochs,
            "max_length": args.max_length,
            "seed": seed,
            "run_number": run_number,
            "num_runs": args.num_runs,
        },
        reinit=True,
    )

    run_metrics = RunMetrics(seed=seed)

    print(f"\n{'=' * 60}")
    print(
        f"Training with {init_type.upper()} initialization (Run {run_number + 1}/{args.num_runs}, Seed {seed})"
    )
    print("Using Unsloth for memory-efficient training")
    print(f"{'=' * 60}\n")

    model, _ = FastLanguageModel.from_pretrained(
        model_name=MODEL_CONFIG["hf_name"],
        max_seq_length=args.max_length,
        dtype=None,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=MODEL_CONFIG["target_modules"],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
    )

    model = reinitialize_lora_weights(model, init_type, args.lora_alpha, args.lora_r)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)"
    )

    log_lora_weight_stats(model, 0, init_type)

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=lambda batch: collate_fn_with_padding(
            batch, pad_token_id=pad_token_id
        ),
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=lambda batch: collate_fn_with_padding(
            batch, pad_token_id=pad_token_id
        ),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    num_training_steps = (
        len(train_dataloader) * args.num_epochs // args.grad_accum_steps
    )
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    global_step = 0

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0

        progress_bar = tqdm(
            train_dataloader, desc=f"Epoch {epoch + 1}/{args.num_epochs}"
        )

        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum_steps
            loss.backward()

            epoch_loss += outputs.loss.item()
            num_batches += 1

            if (step + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                run_metrics.train_losses.append(outputs.loss.item())

                wandb.log(
                    {
                        "train/loss": outputs.loss.item(),
                        "train/learning_rate": scheduler.get_last_lr()[0],
                        "train/epoch": epoch + (step + 1) / len(train_dataloader),
                        "train/global_step": global_step,
                    }
                )

                progress_bar.set_postfix({"loss": outputs.loss.item()})
            if step % EVAL_STEPS == 0:
                eval_metrics = compute_metrics(model, eval_dataloader, device)
                model.train()
                wandb.log(
                    {
                        "eval/loss": eval_metrics["eval_loss"],
                        "eval/perplexity": eval_metrics["perplexity"],
                        "eval/epoch": epoch + (step + 1) / len(train_dataloader),
                        "eval/global_step": global_step,
                    }
                )

        avg_epoch_loss = epoch_loss / num_batches
        eval_metrics = compute_metrics(model, eval_dataloader, device)

        run_metrics.eval_losses.append(eval_metrics["eval_loss"])
        run_metrics.perplexities.append(eval_metrics["perplexity"])

        print(f"\nEpoch {epoch + 1} Results:")
        print(f"  Train Loss: {avg_epoch_loss:.4f}")
        print(f"  Eval Loss: {eval_metrics['eval_loss']:.4f}")
        print(f"  Perplexity: {eval_metrics['perplexity']:.2f}")

        wandb.log(
            {
                "eval/loss": eval_metrics["eval_loss"],
                "eval/perplexity": eval_metrics["perplexity"],
                "train/epoch_loss": avg_epoch_loss,
                "epoch": epoch + 1,
            }
        )

        log_lora_weight_stats(model, epoch + 1, init_type)

        if eval_metrics["eval_loss"] < run_metrics.best_eval_loss:
            run_metrics.best_eval_loss = eval_metrics["eval_loss"]
            run_metrics.best_perplexity = eval_metrics["perplexity"]

    run_metrics.final_train_loss = avg_epoch_loss

    wandb.summary["best_eval_loss"] = run_metrics.best_eval_loss
    wandb.summary["best_perplexity"] = run_metrics.best_perplexity
    wandb.summary["init_type"] = init_type
    wandb.summary["seed"] = seed
    wandb.summary["run_number"] = run_number

    wandb.finish()

    del model
    torch.cuda.empty_cache()

    return run_metrics


def log_lora_weight_stats(model, step, init_type):
    """Log LoRA weight statistics to wandb."""
    a_norms = []
    b_norms = []
    a_stds = []
    b_stds = []

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            for adapter_name in module.lora_A.keys():
                lora_A = module.lora_A[adapter_name].weight.data
                lora_B = module.lora_B[adapter_name].weight.data

                a_norms.append(lora_A.norm().item())
                b_norms.append(lora_B.norm().item())
                a_stds.append(lora_A.std().item())
                b_stds.append(lora_B.std().item())

    if a_norms:
        wandb.log(
            {
                "weights/lora_A_mean_norm": sum(a_norms) / len(a_norms),
                "weights/lora_B_mean_norm": sum(b_norms) / len(b_norms),
                "weights/lora_A_mean_std": sum(a_stds) / len(a_stds),
                "weights/lora_B_mean_std": sum(b_stds) / len(b_stds),
                "weights/step": step,
            }
        )


def log_aggregated_results(init_type: str, stats: dict, args):
    """Log aggregated results for all runs to wandb as a summary run."""
    run_name = f"lora-{init_type}-init-r{args.lora_r}-AGGREGATE-unsloth"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            "model_name": MODEL_CONFIG["hf_name"],
            "init_type": init_type,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "num_runs": args.num_runs,
            "base_seed": args.seed,
            "is_aggregate": True,
        },
        reinit=True,
    )

    for step, (mean, std) in enumerate(
        zip(stats["train_loss_mean"], stats["train_loss_std"])
    ):
        wandb.log(
            {
                "aggregate/train_loss_mean": mean,
                "aggregate/train_loss_std": std,
                "aggregate/train_loss_upper": mean + std,
                "aggregate/train_loss_lower": mean - std,
                "aggregate/global_step": step,
            }
        )

    for epoch, (loss_mean, loss_std, ppl_mean, ppl_std) in enumerate(
        zip(
            stats["eval_loss_mean"],
            stats["eval_loss_std"],
            stats["perplexity_mean"],
            stats["perplexity_std"],
        )
    ):
        wandb.log(
            {
                "aggregate/eval_loss_mean": loss_mean,
                "aggregate/eval_loss_std": loss_std,
                "aggregate/perplexity_mean": ppl_mean,
                "aggregate/perplexity_std": ppl_std,
                "aggregate/epoch": epoch + 1,
            }
        )

    wandb.summary["best_eval_loss_mean"] = stats["best_eval_loss_mean"]
    wandb.summary["best_eval_loss_std"] = stats["best_eval_loss_std"]
    wandb.summary["best_perplexity_mean"] = stats["best_perplexity_mean"]
    wandb.summary["best_perplexity_std"] = stats["best_perplexity_std"]
    wandb.summary["final_train_loss_mean"] = stats["final_train_loss_mean"]
    wandb.summary["final_train_loss_std"] = stats["final_train_loss_std"]
    wandb.summary["num_runs"] = args.num_runs
    wandb.summary["is_aggregate"] = True

    wandb.finish()


def main():
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    print(f"\nModel: {MODEL_CONFIG['hf_name']}")
    print(f"  Target modules: {MODEL_CONFIG['target_modules']}")

    print(f"\nDataset: {DATASET_CONFIG['name']}")

    seeds = [args.seed + i for i in range(args.num_runs)]
    print(f"\nRunning {args.num_runs} experiments with seeds: {seeds}")

    print(f"\nLoading tokenizer: {MODEL_CONFIG['hf_name']}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CONFIG["hf_name"], trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"\nPreparing dataset: {DATASET_CONFIG['name']}")
    train_dataset = prepare_dataset(tokenizer, args.max_length, split="train")
    eval_dataset = prepare_dataset(tokenizer, args.max_length, split="test")

    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")

    results: dict[str, list[RunMetrics]] = {}
    aggregate_stats: dict[str, dict] = {}

    for init_type in ["standard", "reversed", "orthogonal_qr", "orthogonal_eye"]:
        if args.init_strategy not in [init_type, "both"]:
            continue

        print(f"\n{'#' * 70}")
        print(
            f"# Running {args.num_runs} experiments for {init_type.upper()} initialization"
        )
        print(f"{'#' * 70}")

        results[init_type] = []

        for run_idx, seed in enumerate(seeds):
            run_metrics = train_with_init(
                init_type,
                args,
                tokenizer,
                train_dataset,
                eval_dataset,
                device,
                seed=seed,
                run_number=run_idx,
            )
            results[init_type].append(run_metrics)

        aggregate_stats[init_type] = compute_aggregate_stats(results[init_type])
        log_aggregated_results(init_type, aggregate_stats[init_type], args)

    print("\n" + "=" * 70)
    print("FINAL COMPARISON (Mean ± Std over {} runs)".format(args.num_runs))
    print("=" * 70)

    for init_type, stats in aggregate_stats.items():
        print(f"\n  {init_type.upper()} initialization:")
        print(
            f"    Best Eval Loss:    {stats['best_eval_loss_mean']:.4f} ± {stats['best_eval_loss_std']:.4f}"
        )
        print(
            f"    Best Perplexity:   {stats['best_perplexity_mean']:.2f} ± {stats['best_perplexity_std']:.2f}"
        )
        print(
            f"    Final Train Loss:  {stats['final_train_loss_mean']:.4f} ± {stats['final_train_loss_std']:.4f}"
        )


if __name__ == "__main__":
    main()
