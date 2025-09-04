import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import os
import random
import torch.nn.functional as F
import torch.nn as nn

def test_metric_confusion_matrix(model, dataloader, device, classes, output_folder="output", fname="confusion_matrix.png"):
    y_gt, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            x, y = data
            x, y = x.to(device), y.to(device)
            yp = model(x)
            y_pred.append(yp.argmax(1))
            y_gt.append(y)
    # Concatenate all predictions and ground truths
    y_pred_t = torch.cat(y_pred)
    y_gt_t = torch.cat(y_gt)
    # Print accuracy
    accuracy = (y_pred_t == y_gt_t).sum().item() / len(y_gt_t)
    print(f'Accuracy: {accuracy:.4f}')
    # Confusion matrix
    cm = metrics.confusion_matrix(y_gt_t.cpu(), y_pred_t.cpu())
    # Normalize
    cmn = cm.astype(np.float32)
    cmn /= np.sum(cmn, axis=1, keepdims=True)
    cmn = (100 * cmn).astype(np.int32)
    # Plot and save
    disp = metrics.ConfusionMatrixDisplay(cmn, display_labels=classes)
    disp.plot()
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, fname)
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion matrix plot saved to {output_path}")
    
def plot_logit_and_softmax_true(model, testloader, classes, temperature=1.0, output_folder="output"):
    
    for data in testloader:
        x, y = data
        break
    
    k = random.randint(0, x.shape[0] - 1)
    gt_idx = y[k].item()
    output = model(x.cuda())
    pred_idx = output[k].argmax().item()
    gt_label = classes[gt_idx]
    pred_label = classes[pred_idx]

    print(f'GT: {gt_idx}, {gt_label}')
    print(f'Prediction: {pred_idx}, {pred_label}')

    os.makedirs(output_folder, exist_ok=True)

    # Save logits plot
    plt.bar(np.arange(output.shape[1]), output[k].detach().cpu().numpy())
    plt.title('True_Logit')
    logit_path = f'{output_folder}/True_logit_GT_{gt_label}_Pred_{pred_label}.png'
    plt.savefig(logit_path)
    plt.close()

    # Save softmax plot
    s = F.softmax(output / temperature, dim=1)
    plt.bar(np.arange(output.shape[1]), s[k].detach().cpu().numpy())
    plt.title(f'True_Softmax T={temperature}')
    softmax_path = f'{output_folder}/True_softmax_GT_{gt_label}_Pred_{pred_label}.png'
    plt.savefig(softmax_path)
    plt.close()

    # Save input image plot with legend
    img = x[k].cpu().permute(1, 2, 0)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'GT: {gt_label} | Pred: {pred_label}')
    image_path = f'{output_folder}/True_image_GT_{gt_label}_Pred_{pred_label}.png'
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    print(f"Saved: {logit_path}")
    print(f"Saved: {softmax_path}")
    print(f"Saved: {image_path}")
    
def plot_logit_and_softmax_fake(model, fakeloader, classes, temperature=1.0, output_folder="output"):
    
    for data in fakeloader:
        x, _ = data
        break
    
    k = random.randint(0, x.shape[0] - 1)
    output = model(x.cuda())
    pred_idx = output[k].argmax().item()
    pred_label = classes[pred_idx]

    print(f'Prediction: {pred_idx}, {pred_label}')

    os.makedirs(output_folder, exist_ok=True)

    # Save logits plot
    plt.bar(np.arange(output.shape[1]), output[k].detach().cpu().numpy())
    plt.title('Fake_Logit')
    logit_path = f'{output_folder}/Fake_Pred_{pred_label}.png'
    plt.savefig(logit_path)
    plt.close()

    # Save softmax plot
    s = F.softmax(output / temperature, dim=1)
    plt.bar(np.arange(output.shape[1]), s[k].detach().cpu().numpy())
    plt.title(f'Fake_Softmax T={temperature}')
    softmax_path = f'{output_folder}/softmax_Pred_{pred_label}.png'
    plt.savefig(softmax_path)
    plt.close()

    # Save input image plot with legend
    img = x[k].cpu().permute(1, 2, 0)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Pred: {pred_label}')
    image_path = f'{output_folder}/Fake_image_Pred_{pred_label}.png'
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    print(f"Saved: {logit_path}")
    print(f"Saved: {softmax_path}")
    print(f"Saved: {image_path}")
    
def max_logit(logit):
    s = logit.max(dim=1)[0] #get the max for each element of the batch
    return s

def compute_scores(model, device, data_loader, score_fun):
    scores = []
    with torch.no_grad():
        for data in data_loader:
            x, y = data
            output = model(x.to(device))
            s = score_fun(output)
            scores.append(s)
        scores_t = torch.cat(scores)
        return scores_t
    

def plot_distribution(scores_test, scores_fake, model_name, output_folder="output"):
    os.makedirs(output_folder, exist_ok=True)

    # Plot line plot
    plt.figure()
    plt.plot(sorted(scores_test.cpu()), label='test')
    plt.plot(sorted(scores_fake.cpu()), label='fake')
    plt.legend()
    sorted_plot_path = f"{output_folder}/line_plot_scores_" + model_name + ".png"
    plt.savefig(sorted_plot_path)
    plt.close()

    # Plot histogram of scores
    plt.figure()
    plt.hist(scores_test.cpu(), density=True, alpha=0.5, bins=25, label='test')
    plt.hist(scores_fake.cpu(), density=True, alpha=0.5, bins=25, label='fake')
    plt.legend()
    hist_plot_path = f"{output_folder}/histogram_scores_" + model_name + ".png"
    plt.savefig(hist_plot_path)
    plt.close()

    print(f"Saved sorted plot to {sorted_plot_path}")
    print(f"Saved histogram plot to {hist_plot_path}")
    
def compute_scores_autoencoder(model, device, data_loader):
    loss_fn = nn.MSELoss(reduction='none')
    scores = []
    model.eval()
    with torch.no_grad():
        for data in data_loader:
            x, _ = data
            x = x.to(device)
            _, xr = model(x) 
            loss = loss_fn(xr, x)  # reconstruction loss per element
            score = loss.mean(dim=[1, 2, 3])  # mean error per sample
            scores.append(-score)  # negative MSE
    scores_t = torch.cat(scores)
    return scores_t

def plot_metrics(scores_test, scores_fake, model_name, output_folder="output"):
    os.makedirs(output_folder, exist_ok=True)

    y_pred = torch.cat((scores_test, scores_fake))
    y_true = torch.cat((
        torch.ones_like(scores_test),
        torch.zeros_like(scores_fake)
    ))

    # ROC Curve
    plt.figure()
    metrics.RocCurveDisplay.from_predictions(y_true.cpu(), y_pred.cpu())
    plt.title(f'ROC Curve - {model_name}')
    roc_path = os.path.join(output_folder, f'roc_curve_{model_name}.png')
    plt.savefig(roc_path)
    plt.close()

    # Precision-Recall Curve
    plt.figure()
    metrics.PrecisionRecallDisplay.from_predictions(y_true.cpu(), y_pred.cpu())
    plt.title(f'Precision-Recall Curve - {model_name}')
    pr_path = os.path.join(output_folder, f'precision_recall_curve_{model_name}.png')
    plt.savefig(pr_path)
    plt.close()

    print(f"Saved ROC curve plot to {roc_path}")
    print(f"Saved Precision-Recall curve plot to {pr_path}")