# A×0 $\neq$ 0×B: The Hidden Impact of Initialization in LoRA

This repository contains experiments comparing two different LoRA (Low-Rank Adaptation) initialization strategies for fine-tuning large language models: initializing either $A$ or $B$ with zero and the other randomly. The implementation uses [Unsloth](https://github.com/unslothai/unsloth) for memory-efficient training and Weights & Biases for experiment tracking.

## Overview

LoRA is a popular parameter-efficient fine-tuning method that adds trainable low-rank matrices to pre-trained model weights. This project compares two initialization strategies:

1. **Standard initialization**: Matrix A is randomly initialized, Matrix B is zero-initialized
2. **Reversed initialization**: Matrix A is zero-initialized, Matrix B is randomly initialized

The experiments are run multiple times with different random seeds to provide statistically robust comparisons.

## Results

For a detailed analysis of the experimental results, see the blog post: [The Hidden Impact of Initialization in LoRA](https://aweers.de/blog/2025/lora-init/)

![Plot of both initialization schemes with learning rates on x-axis and final evaluation loss on y-axis](/results/lr_vs_final_eval_loss.png)

![Plot of both initialization schemes with learning rates on x-axis and final evaluation loss on y-axis (zoomed in)](/results/lr_vs_final_eval_loss_filtered.png)

![Loss over steps for learning rate where Init A is still stable, but Init B is already unstable](/results/loss_curves_lr_1e03_eval_loss.png)
