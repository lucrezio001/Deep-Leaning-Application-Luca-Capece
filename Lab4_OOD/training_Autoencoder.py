import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import wandb
from tqdm import tqdm
import os
from model import Autoencoder

# output folder
os.makedirs("Save", exist_ok=True)


def train_one_epoch_ae(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    # no label needed
    for x, _ in tqdm(dataloader, desc="Training AE", leave=False):
        x = x.to(device)
        optimizer.zero_grad()
        _, outputs = model(x)
        loss = loss_fn(outputs, x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(dataloader.dataset)


def evaluate_ae(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, _ in tqdm(dataloader, desc="Evaluating AE", leave=False):
            x = x.to(device)
            _, outputs = model(x)
            loss = loss_fn(outputs, x)
            total_loss += loss.item() * x.size(0)
    return total_loss / len(dataloader.dataset)


def train_autoencoder_model(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"])

    model = Autoencoder().to(device)

    run = wandb.init(name=config["run_name"]+"_Autoencoder", project=config["project_name"], config=config, reinit=True)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = nn.MSELoss()

    for epoch in tqdm(range(config["epochs"]), desc="Epochs AE"):
        train_loss = train_one_epoch_ae(model, train_loader, optimizer, loss_fn, device)
        val_loss = evaluate_ae(model, val_loader, loss_fn, device)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        tqdm.write(f"Epoch {epoch+1}/{config['epochs']} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    torch.save(model.state_dict(), "Save/Cifar10_Autoencoder.pth")
    run.finish()

    return model


if __name__ == "__main__":
    config = {
        "project_name": "Lab4_OOD",
        "data": "CIFAR10",
        "run_name": "OOD_Autoencoder",
        "batch_size": 128,
        "lr": 0.001,
        "epochs": 40,
    }

    train_autoencoder_model(config)