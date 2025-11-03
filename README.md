# Parameter-Efficient Fine-Tuning with LORA for AG News Classification

This repository contains the code for a Deep Learning project focused on building a highly efficient text classifier for the AG News dataset.

The primary goal was to fine-tune a `roberta-base` model while strictly keeping the number of trainable parameters under **1 million**. We achieved this by implementing **Low-Rank Adaptation (LoRA)** and **teacher-student distillation**.

Our final model achieves **94.53% validation accuracy** with only **925,444 trainable parameters**, demonstrating that parameter-efficient techniques can match the performance of full-model fine-tuning.

## Key Methodology

* **Base Model:** `roberta-base`
* **Core Technique:** Low-Rank Adaptation (LoRA)
* **Enhancement:** Teacher-Student Distillation, using a fully fine-tuned `roberta-base` as the teacher model.
* **Optimizer:** AdamW with a polynomial learning rate schedule.

## Performance & Final Configuration

Our final configuration balanced a low parameter count with high performance by targeting the most critical components of the transformer architecture.

| Metric | Value |
| :--- | :--- |
| **Validation Accuracy** | 94.53% |
| **Trainable Parameters** | 925,444 |
| **% of Full Model** | ~0.74% |
| **LoRA Rank (r)** | 6 |
| **LoRA Target Modules** | `query`, `key`, `value` |
| **LoRA Scaling ($\alpha$)** | 16 |
