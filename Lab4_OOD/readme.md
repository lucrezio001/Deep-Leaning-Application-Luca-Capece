# Laboratory 4: Adversarial Learning and OOD Detection

## Models

- **ResCNN:** Residual Convolutional Neural Network based on ResNet18 blocks for robust image classification (Same architecture as Lab1_CNN).
- **Autoencoder:** Unsupervised model used for OOD detection.


## Datasets

- [**CIFAR10:**](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html) Main training and evaluation dataset for all supervised and unsupervised experiments.
- [**FakeData:**](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.FakeData.html) Synthetic unlabeled data for OOD detection demonstration.

## Pipeline Overview

1. **OOD Detection and Performance Evaluation Using CNN**
    - Confusion matrix to check CNN performance on CIFAR10
        ![confusion_matrix](output/confusion_matrix.png)
    - Visualize model predictions for a sample through logit and softmax outputs on CIFAR10:
        ![image](output/True_image_GT_truck_Pred_truck.png)
        ![logit](output/True_logit_GT_truck_Pred_truck.png)
        ![softmax](output/True_softmax_GT_truck_Pred_truck.png)
    - Visualize model predictions for a sample through logit and softmax outputs on FakeData:
        ![image](output/Fake_image_Pred_horse.png)
        ![logit](output/Fake_Pred_horse.png)
        ![softmax](output/Fake_softmax_Pred_horse.png)
    - Distribution of dataset for CNN:
        ![line_distr](output/line_plot_scores_CNN.png)
        ![hist_distr](output/histogram_scores_CNN.png)

2. **OOD Detection and Performance Evaluation Using Autoencoder**
    - Distribution of dataset for CNN:
        ![line_distr](output/line_plot_scores_Autoencoder.png)
        ![hist_distr](output/histogram_scores_Autoencoder.png)

3. **OOD Detection Metrics for CNN & Autoencoder**
    - Precision Recall and ROC curve for CNN:
        ![P_R](output/precision_recall_curve_CNN.png)
        ![ROC](output/roc_curve_CNN.png)
    - Precision Recall and ROC curve for Autoencoder:
        ![P_R](output/precision_recall_curve_CNN.png)
        ![ROC](output/roc_curve_CNN.png)

4. **Enhancing Robustness to Adversarial Attack**

    - Fast Gradient Sign Method (FGSM) perturbs samples in the direction of the gradient with respect to the input $\mathbf{x}$:
            $$ \boldsymbol{\eta}(\mathbf{x}) = \varepsilon \mathrm{sign}(\nabla_{\mathbf{x}} \mathcal{L}(\boldsymbol{\theta}, \mathbf{x}, y)) ) $$

        where $\varepsilon$ is the attack budget controlling perturbation magnitude.

    - Baseline Untargeted FGSM Attack
        - Original image
            ![img](output/untargeted_baseline/original_image.png)
        - Perturbation
            ![perturbation](output/untargeted_baseline/perturbation_image_3_steps.png)
        - Scale of the perturbation
            ![hist](output/untargeted_baseline/perturbation_histogram_3_steps.png)
        - New adversarial image
            ![adv_img](output/untargeted_baseline/adversarial_image_3_steps.png)

    > **Note:** Attack budget $\varepsilon = 6/255$
- Untargeted FGSM Attack after Untargeted Adversarial training
    - [img](output/untargeted_trained_on_adv_sample/original_image.png)
    - [perturbation](output/untargeted_trained_on_adv_sample/perturbation_image_6_steps.png)
    - [hist](output/untargeted_trained_on_adv_sample/perturbation_histogram_6_steps.png)
    - [adv_img](output/untargeted_trained_on_adv_sample/adversarial_image_6_steps.png)

    > **Note:** More Attack budget spent $\varepsilon = 12/255$

### Untargeted attack success rates
$$
\text{Untargeted Attack Success Rate} = \frac{\# \text{ of adversarial samples classified differently from ground truth}}{\text{total adversarial samples}} \times 100
$$

| Metric | Value (%) |
| :-- | :-- |
| Untargeted Attack Success Rate | 75.68 |
| Untargeted Attack Success Rate Robust | 33.14 |


### Targeted Adversarial Training

- Baseline Targeted FGSM Attack class Cat (Same class used for robust training)
        - Original image
            ![img](output/targeted_baseline_same_class/original_image.png)
        - Perturbation
            ![perturbation](output/targeted_baseline_same_class/perturbation_image_4_steps.png)
        - Scale of the perturbation
            ![hist](output/targeted_baseline_same_class/perturbation_histogram_4_steps.png)
        - New adversarial image
            ![adv_img](output/targeted_baseline_same_class/adversarial_image_4_steps.png)

    > **Note:** Attack budget $\varepsilon = 8/255$

- Baseline Targeted FGSM Attack class Horse (Different class used for robust training)
        - Original image
            ![img](output/targeted_baseline_different_class/original_image.png)
        - Perturbation
            ![perturbation](output/targeted_baseline_different_class/perturbation_image_3_steps.png)
        - Scale of the perturbation
            ![hist](output/targeted_baseline_different_class/perturbation_histogram_3_steps.png)
        - New adversarial image
            ![adv_img](output/targeted_baseline_different_class/adversarial_image_3_steps.png)

    > **Note:** Attack budget $\varepsilon = 6/255$

- Targeted FGSM Attack class "cat", after Targeted Adversarial training on "cat" class
        - Original image
            ![img](output/targeted_trained_baseline_same_class/original_image.png)
        - Perturbation
            ![perturbation](output/targeted_trained_baseline_same_class/perturbation_image_4_steps.png)
        - Scale of the perturbation
            ![hist](output/targeted_trained_baseline_same_class/perturbation_histogram_4_steps.png)
        - New adversarial image
            ![adv_img](output/targeted_trained_baseline_same_class/adversarial_image_4_steps.png)

    > **Note:** Same Attack budget $\varepsilon = 8/255$ it worked?

- Targeted FGSM Attack class "horse", after Targeted Adversarial training on "cat" class
        - Original image
            ![img](output/targeted_baseline_different_class/original_image.png)
        - Perturbation
            ![perturbation](output/targeted_baseline_different_class/perturbation_image_3_steps.png)
        - Scale of the perturbation
            ![hist](output/targeted_baseline_different_class/perturbation_histogram_3_steps.png)
        - New adversarial image
            ![adv_img](output/targeted_baseline_different_class/adversarial_image_3_steps.png)

    > **Note:** Same Attack budget $\varepsilon = 6/255$ it worked?


### Targeted attack success rates
$$
\text{Targeted Attack Success Rate} = \frac{\# \text{ of adversarial samples classified as target class}}{\text{total adversarial samples}} \times 100
$$

**Targeted Attack Success Rate Before Training**

| Class | Success Rate (%) |
| :-- | :-- |
| Airplane | 31.84 |
| Automobile | 26.51 |
| Bird | 52.16 |
| Cat | 50.36 |
| Deer | 36.55 |
| Dog | 21.69 |
| Frog | 17.51 |
| Horse | 46.45 |
| Ship | 26.37 |
| Truck | 16.05 |

**Targeted Attack Success Rate After Training**

| Class | Success Rate (%) |
| :-- | :-- |
| Airplane | 14.66 |
| Automobile | 15.27 |
| Bird | 17.89 |
| Cat | 13.92 |
| Deer | 25.99 |
| Dog | 24.21 |
| Frog | 18.55 |
| Horse | 16.91 |
| Ship | 13.70 |
| Truck | 12.49 |

For the cat and horse classes generally we have lower targeted attack success rate, however for some classes like dog and frog we can se higher targeted attack success rate.

So overall targeted training work better than untargeted?

**Targeted Attack Success Rate After Training**

| Metric | Value (%) |
| :-- | :-- |
| Untargeted Attack Success Rate | 75.68 |
| Untargeted Attack Success Rate Robust | 33.14 |
| Targeted Attack Success Rate Robust | 41.26 |


## Output Folder Structure

All plots and images are automatically organized as follows:

```
output/
│
├── untargeted_baseline/
│   ├── original_image.png                     # Clean input
│   ├── adversarial_image_*_steps.png          # FGSM adversarial sample (untargeted, baseline)
│   ├── perturbation_image_*_steps.png         # Perturbation visualization added to image
│   ├── perturbation_histogram_*_steps.png     # Perturbation histogram
│
├── untargeted_trained_on_adv_sample/
│   ├── original_image.png                     # Clean input
│   ├── adversarial_image_*_steps.png          # Adversarial sample (untargeted, post-training)
│   ├── perturbation_image_*_steps.png         # Perturbation visualization added to image (post-training)
│   ├── perturbation_histogram_*_steps.png     # Perturbation histogram (post-training)
│
├── targeted_baseline_same_class/
│   ├── original_image.png                     # Clean input (car → cat)
│   ├── adversarial_image_*_steps.png          # FGSM adversarial sample (targeted, baseline)
│   ├── perturbation_image_*_steps.png         # Perturbation visualization added to image
│   ├── perturbation_histogram_*_steps.png     # Perturbation histogram
│
├── targeted_baseline_different_class/
│   ├── original_image.png                     # Clean input (car → horse)
│   ├── adversarial_image_*_steps.png          # FGSM adversarial sample (targeted, baseline)
│   ├── perturbation_image_*_steps.png         # Perturbation visualization added to image
│   ├── perturbation_histogram_*_steps.png     # Perturbation histogram
│
├── targeted_trained_same_different_class/
│   ├── original_image.png                     # Clean input (car → cat)
│   ├── adversarial_image_*_steps.png          # FGSM adversarial sample (targeted, post-training target cat)
│   ├── perturbation_image_*_steps.png         # Perturbation visualization added to image
│   ├── perturbation_histogram_*_steps.png     # Perturbation histogram
│
├── targeted_trained_baseline_different_class/
│   ├── original_image.png                     # Clean input (car → horse)
│   ├── adversarial_image_*_steps.png          # FGSM adversarial sample (targeted, post-training target cat)
│   ├── perturbation_image_*_steps.png         # Perturbation visualization added to image
│   ├── perturbation_histogram_*_steps.png     # Perturbation histogram
│
├── confusion_matrix.png                   # Confusion matrix for CNN
├── Fake_image_horse.png                   # Example fake image
├── Fake_Pred_horse.png                    # Example fake logits result
├── Fake_softmax_Pred_horse.png            # Example fake softmax scores
├── histogram_scores_CNN.png               # Dataset histogram distribution for CNN
├── histogram_scores_Autoencoder.png       # Dataset histogram distribution for Autoencoder
├── line_plot_scores_CNN.png               # Dataset line plot distribution for CNN 
├── line_plot_scores_Autoencoder.png       # Dataset line plot distribution for Autoencoder
├── precision_recall_curve_CNN.png         # CNN model precision-recall curve
├── precision_recall_curve_Autoencoder.png # Autoencoder precision-recall curve
├── roc_curve_CNN.png                      # CNN ROC curve
├── roc_curve_Autoencoder.png              # Autoencoder ROC curve
├── True_image_GT_truck_Pred_truck.png     # Example true image
├── True_logit_GT_truck_Pred_truck.png     # Example true logits result
├── True_softmax_GT_truck_Pred_truck.png   # Example true softmax scores
Fake_image_horse.png

```


## How to Run

1. **Clone repo**

```bash
    git clone https://github.com/lucrezio001/Deep-Leaning-Application-Luca-Capece.git
    cd Deep-Leaning-Application-Luca-Capece/Lab4_OOD
```
2. **install dependencies**
```bash
    pip install -r requirements.txt
```
3. **Training**
    - Before running "Lab4_OOD.py" CNN and Autoencoder model are Required
        - Checkpoint are found in Save folder (Also training can be found on W&B)
        [![W&B Project](https://img.shields.io/badge/W%26B-Project-lightgrey?logo=wandb)](https://wandb.ai/lucacapece007-universit-di-firenze/Lab4_OOD?nw=nwuserlucacapece007)
    - If needed you can also train from scratch CNN and autoencoder from "training_CNN.py" and "training_Autoencoder.py"
```bash
python training_CNN.py
```
```bash
python training_Autoencoder.py
```
4. **Visualize Results**
    - Run `Lab4_OOD.py` to perform all the content described above.
```bash
python Lab4_OOD.py
```
    - All outputs and visualizations are saved in the `output` folder following the directory structure described above.

***

## References

- Liang, S., Li, Y., & Srikant, R. (2018). Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks. International Conference on Learning Representations (ICLR).  https://arxiv.org/abs/1706.02690
- He, K., Zhang, X., Ren, S., \& Sun, J. (2015). Deep Residual Learning for Image Recognition. https://doi.org/10.48550/arXiv.1512.03385

***
