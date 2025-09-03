# Transformer Lab

This repository contains deep learning experiments focused on transformer-based models, including fine-tuning and evaluation with classification tasks.

## Exercises Overview

- **Exercise 1 \& 2:** Implemented in `transformer_1-2.ipynb`.
- **Exercise 3.2:** Scripts in `transformer_3_2_Lora_fine_tuned.py` and `transformer_3_2_metrics.py`—both utilize `utility.py` functions.


## Models

- **CLIP (Contrastive Language-Image Pre-training):**
    - Used as base model, augmented for classification tasks.
    - Fine-tuned using the LoRA (Low-Rank Adaptation) technique for improved parameter efficiency.
- **LoRA Fine-Tuned Models:**
    - Three modes:
        - *Text*: Only the text encoder is trained (visual encoder frozen).
        - *Vision*: Only the visual encoder is trained (text encoder frozen).
        - *Hybrid*: Both encoders are trained.


## Dataset

Experiments use the [**Beans**](https://huggingface.co/datasets/AI-Lab-Makerere/beans) dataset from Makerere AI Lab:

- **Description:**
    - Contains labeled bean crop images (healthy, angular leaf spot, bean rust).
    - Small dataset but usefull for testing all CLIP fine-tuning combination.


## Experiment Tracking with Weights \& Biases

All experiments, metrics, and visualizations are tracked in W\&B:

- [Lab3_transformer_3-2_clip Project](https://wandb.ai/lucacapece007-universit-di-firenze/Lab3_transformer_3-2_clip?nw=nwuserlucacapece007)


## Fine-tuning \& Evaluation Workflow

- **Fine-tuning CLIP with LoRA:**
    - Execute `transformer_3_2_Lora_fine_tuned.py` to add a classification head and train in the selected mode (edit `Config/default.yaml`).
- **Evaluating Metrics:**
    - Run `transformer_3_2_metrics.py` for F1, precision, recall, and accuracy. Reports for both zero-shot and fine-tuned models.
    - To evaluate a custom checkpoint: rename its folder to `best_{mode}` (e.g., `beans_text_vision/best_hybrid`).


## Configuration

Edit all hyperparameters and mode settings in `Config/default.yaml`:

- *Modes:*
    - `text`: Fine-tune text encoder only; vision encoder frozen.
    - `vision`: Fine-tune vision encoder only; text encoder frozen.
    - `hybrid`: Fine-tune both vision and text encoders.


## Installation

1. Clone the repository:

```
git clone https://github.com/lucrezio001/Deep-Leaning-Application-Luca-Capece.git
cd Deep-Leaning-Application-Luca-Capece/Lab3_Transformer
```

2. Install dependencies:

```
pip install -r requirements.txt
```


## Results Table

Below is a template for reporting test metrics across evaluated modes:


| Model Mode | F1 | Precision | Recall | Accuracy |
| :-- | :-- | :-- | :-- | :-- |
| Zero-shot |  |  |  |  |
| Text |  |  |  |  |
| Vision |  |  |  |  |
| Hybrid |  |  |  |  |

_Fill in results after running `transformer_3_2_metrics.py`._

## Bibliography

- Radford, A., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
- Hu, E., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Makerere AI Lab. Bean disease dataset. [Dataset Info](https://huggingface.co/datasets/AI-Lab-Makerere/beans)[^2][^1]
