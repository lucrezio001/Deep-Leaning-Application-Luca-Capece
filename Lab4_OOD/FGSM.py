import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from torchvision import transforms
import wandb
from tqdm import tqdm

class NormalizeInverse(transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())

def fgsm_attack_save_plots(model, testloader, device, testset_classes,
                          targeted_attack=False, target_label=None, eps=2/255,
                          output_folder="output", sample_id=0):
    os.makedirs(output_folder, exist_ok=True)
    
    for data in testloader:
        x, y = data
        break

    inv = NormalizeInverse((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    loss_fn = nn.CrossEntropyLoss()
    model.eval()

    x = x[sample_id].to(device)
    y = y[sample_id].to(device)

    x = x.unsqueeze(0)
    y = y.unsqueeze(0)
    x.requires_grad = True

    before = x.clone()
    output = model(x)
    pred_class = output.argmax().item()

    #Save original image
    img = inv(x[0])
    plt.imshow(img.permute(1, 2, 0).detach().cpu())
    plt.title(testset_classes[pred_class])
    plt.savefig(os.path.join(output_folder, "original_image.png"))
    plt.close()

    if pred_class != y.item() or y.item() == target_label:
        print('Classifier is already wrong or target label same as GT!')
        return

    done = False
    n = 0
    if targeted_attack:
        target = torch.tensor([target_label]).to(device)
        print(f'Target: {testset_classes[target_label]}')

    print('Attack!')

    while not done:
        x.retain_grad()
        output = model(x)

        model.zero_grad()
        if targeted_attack:
            yt = target
        else:
            yt = y
            
        loss = loss_fn(output, yt)
        loss.backward()

        if targeted_attack:
            x = x - eps * torch.sign(x.grad)
        else:
            x = x + eps * torch.sign(x.grad)

        n += 1

        pred_class = output.argmax().item()
        print(pred_class, y.item())

        if not targeted_attack and pred_class != y.item():
            print(f'Untargeted attack success! budget: {int(255 * n * eps)}/255')
            done = True

        if targeted_attack and pred_class == target_label:
            print(f'Targeted attack ({testset_classes[pred_class]}) success! budget: {int(255 * n * eps)}/255')
            done = True

    # Save adversarial image
    adv_img = inv(x.squeeze())
    plt.imshow(adv_img.permute(1, 2, 0).detach().cpu())
    plt.title(f"Adversarial: {testset_classes[pred_class]}")
    plt.savefig(os.path.join(output_folder, f"adversarial_image_{n}_steps.png"))
    plt.close()

    # Save perturbation image
    diff = x - before
    diff_img = inv(diff[0])
    plt.imshow(diff_img.permute(1, 2, 0).detach().cpu())
    plt.title('Perturbation')
    plt.savefig(os.path.join(output_folder, f"perturbation_image_{n}_steps.png"))
    plt.close()
        
    # Save histogram of perturbation
    diff_flat = diff.flatten()
    plt.hist(diff_flat.detach().cpu(), bins=50)
    plt.title('Histogram of perturbation')
    plt.savefig(os.path.join(output_folder, f"perturbation_histogram_{n}_steps.png"))
    plt.close()

    print(f"Saved adversarial image, perturbation image, and histogram to {output_folder}")


# FGSM single iteration

def fgsm_attack(model, x, y, device, eps, targeted=False, target_label=None):
    x_adv = x.clone().detach().to(device)
    x_adv.requires_grad = True

    output = model(x_adv)
    loss_fn = nn.CrossEntropyLoss()

    if targeted:
        target = torch.full_like(y, target_label).to(device)
        loss = loss_fn(output, target)
    else:
        loss = loss_fn(output, y)

    model.zero_grad()
    loss.backward()

    if targeted:
        x_adv = x_adv - eps * torch.sign(x_adv.grad) #FGSM targheted
    else:
        x_adv = x_adv + eps * torch.sign(x_adv.grad) #FGSM targheted

    return x_adv.detach()


# Check Attack Success Rate for targeted and untargeted attack

# Targeted Attack Success Rate = (# of x_adv classified as target) / (all x_adv) * 100

# Untargeted Attack Success Rate= (# of x_adv different from ground truth) / (all x_adv) * 100

def attack_success_rate(model, dataloader, device, testset_classes, eps, targeted=False, target_label=None):

    model.eval()
    total = 0
    success = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Adversarial examples
        adv_inputs = fgsm_attack(model, inputs, labels, device, eps, targeted=targeted, target_label=target_label)

        # Prediction
        outputs = model(adv_inputs)
        preds = outputs.argmax(dim=1)

        if targeted:
            # sum if prediction = target_label
            success += (preds == target_label).sum().item()
        else:
            # sum if prediction ≠ ground truth
            success += (preds != labels).sum().item()

        total += inputs.size(0)

    success_rate = 100 * success / total

    if targeted:
        print(f"Targeted Attack Success Rate towards class {testset_classes[target_label]}: {success_rate:.2f}%")
    else:
        print(f"Untargeted Attack Success Rate: {success_rate:.2f}%")

    return success_rate

# Training with adversarial example

def train_with_fgsm_adversarial(model, trainloader, device, num_epochs=10, eps=2/255,
                               targeted=False, target_label=None, lr=0.001,
                               project_name="Lab4_OOD_FGSM", run_name="fgsm_adv_train"):
    # W&B
    run = wandb.init(project=project_name, name=run_name, reinit=True)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        epoch_iterator = tqdm(trainloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for i, (inputs, labels) in enumerate(epoch_iterator):
            inputs, labels = inputs.to(device), labels.to(device)

            adv_inputs = fgsm_attack(model, inputs, labels, device, eps, targeted=targeted, target_label=target_label)
            combined_inputs = torch.cat([inputs, adv_inputs], dim=0)
            combined_labels = torch.cat([labels, labels], dim=0)

            outputs = model(combined_inputs)
            loss = loss_fn(outputs, combined_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 100 == 0:
                avg_loss = running_loss / 100
                epoch_iterator.set_postfix(loss=avg_loss)
                wandb.log({"epoch": epoch + 1, "step": i + 1 + epoch * len(trainloader), "loss": avg_loss})
                running_loss = 0.0

    print("End training")
    run.finish()

def success_rate_each_class(model, testloader,  device, testset_classes, class_dict, eps=2/255):
    tasr_results = {}
    for class_name, target_label in class_dict.items():
        tasr = attack_success_rate(
            model,
            testloader,
            device, 
            testset_classes,
            eps=eps,
            targeted=True,
            target_label=target_label
        )
        tasr_results[class_name] = tasr
    return tasr_results

