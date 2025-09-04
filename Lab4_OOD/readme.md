# Laboratory 4: Adversarial Learning and OOD Detection

## Models

- **ResCNN:** Residual Convolutional Neural Network based on ResNet18 blocks for robust image classification (Same architecture as Lab1_CNN).
- **Autoencoder:** Unsupervised model used for OOD detection.


## Datasets

- [**CIFAR10:**](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html) Main training and evaluation dataset for all supervised and unsupervised experiments.
- [**FakeData:**](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.FakeData.html) Synthetic unlabeled data for OOD detection demonstration.

## Pipeline Overview

1. **OOD Detection and Performance Evaluation**
    - Compute confusion matrix (`confusion_matrix.png`)
    - Visualize model predictions through logit and softmax outputs:
        - Predicted class image (`softmax_Pred_*.png`)
        - True class image (`True_image_GT_*.png`)
    - OOD detection via MaxLogit scores and Autoencoder reconstruction error:
        - `line_plot_scores_CNN.png`, `histogram_scores_CNN.png`, `roc_curve_CNN.png`
        - `line_plot_scores_Autoencoder.png`, `precision_recall_curve_Autoencoder.png`
2. **Adversarial Example Generation (FGSM)**
    - Baseline untargeted FGSM attack:
        - Images and perturbations saved per run (e.g., `output/untargeted_baseline/adversarial_image_*.png`)
    - Adversarial training and post-training attack evaluation:
        - `output/untargeted_trained_on_adv_sample/`
    - Targeted FGSM attacks for:
        - Same class (`output/targeted_baseline_same_class/`)
        - Different class (`output/targeted_baseline_different_class/`)
    - Perturbation image and histogram files document attack progress.
3. **Adversarial Training \& Success Rates**
    - Augment training with adversarial samples (`train_with_fgsm_adversarial`)
    - Evaluate attack success rates, overall and per-class (`attack_success_rate`, `success_rate_each_class`)
    - All outputs and intermediate results are visualized and saved in the `output` directory.

***

## Output Folder Structure

All plots and images are automatically organized as follows:

```
output/
│
├── untargeted_baseline/
│   ├── original_image.png
│   ├── adversarial_image_*_steps.png
│   ├── perturbation_image_*_steps.png
│   ├── perturbation_histogram_*_steps.png
│
├── untargeted_trained_on_adv_sample/
│   ├── original_image.png
│   ├── adversarial_image_*_steps.png
│   ├── perturbation_image_*_steps.png
│   ├── perturbation_histogram_*_steps.png
│   ├── confusion_matrix.png
│   ├── [other OOD/CNN/Ae metric plots]
│
├── targeted_baseline_same_class/
│   ├── original_image.png
│   ├── adversarial_image_*_steps.png
│   ├── perturbation_image_*_steps.png
│   ├── perturbation_histogram_*_steps.png
│
├── targeted_baseline_different_class/
│   ├── original_image.png
│   ├── adversarial_image_*_steps.png
│   ├── perturbation_image_*_steps.png
│   ├── perturbation_histogram_*_steps.png
│
```

*Refer to the images above for examples of actual file layout and output content.*

***

## Example Results

Below are sample experiment outputs (see `/output`):

- **Adversarial Images:**
- **Perturbation Visualizations:**
- **Input and Predictions:**
- **OOD and Adversarial Metrics:**

***

## How to Run

1. **Clone this repo and install dependencies**

```bash
git clone <repo-url>
cd <repo-folder>
pip install -r requirements.txt
```

2. **Train and Evaluate**
    - All scripts and modules are ready for execution via `python main.py` (see code structure above).
    - Outputs are automatically organized under `/output`.
3. **Visualize Results**
    - Explore images, perturbations, and metric plots in the output folders.
    - Track full training and evaluation history via Weights \& Biases.

***

## References

- Goodfellow, I.J., Shlens, J., \& Szegedy, C. (2015). Explaining and harnessing adversarial examples. [arXiv:1412.6572](https://arxiv.org/abs/1412.6572)
- He, K., Zhang, X., Ren, S., \& Sun, J. (2015). Deep Residual Learning for Image Recognition. [arXiv:1512.03385](https://doi.org/10.48550/arXiv.1512.03385)

***
