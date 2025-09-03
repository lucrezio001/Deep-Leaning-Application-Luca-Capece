import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
from models import MLP, ResMLP, PlainCNN, ResCNN
import os

# output folder
os.makedirs("output", exist_ok=True)


# Training function

def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss, correct = 0, 0
    for x, y in tqdm(dataloader, desc="Training", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (outputs.argmax(1) == y).sum().item()
        
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = correct / len(dataloader.dataset)
    return avg_loss, accuracy


def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Evaluating", leave=False):
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = loss_fn(outputs, y)

            total_loss += loss.item() * x.size(0)
            correct += (outputs.argmax(1) == y).sum().item()

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = correct / len(dataloader.dataset)
    return avg_loss, accuracy

# Training Loop

def train_model(config):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    #Dataset selection
    
    if config["data"] == "MNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config["batch_size"])
        
    elif config["data"] == "CIFAR10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config["batch_size"])
    else:
        raise ValueError(f"Select valid data MNIST or CIFAR10")

    # Training loop
    for depth in config["depths"]:
        
        if config["data"] == "MNIST":
            # Model selection for MNIST
            if config["model"] == "MLP":
                model = MLP(input_size=config["input_size"] ,hidden_layers=config["hidden_layers"], depth = depth, output_size = config["output_size"]).to(device)
            elif config["model"] == "ResMLP":
                model = ResMLP(input_size=config["input_size"] ,hidden_layers=config["hidden_layers"], depth = depth, output_size = config["output_size"]).to(device)
                
        elif config["data"] == "CIFAR10":
            # Model selection for CIFAR10
            if config["model"] == "PlainCNN":
                model = PlainCNN(depth=depth, in_channels = config["input_size"], output_size = config["output_size"]).to(device)
            elif config["model"] == "ResCNN":
                model = ResCNN(depth=depth, in_channels = config["input_size"], output_size = config["output_size"]).to(device)

        # W&B
        run = wandb.init(name= config["run_name"] + "_depth_" + str(depth), project= config["project_name"], config = config, reinit=True)
        
        
        optimizer = optim.Adam(model.parameters(), lr=config["lr"])
        loss_fn = nn.CrossEntropyLoss()
        
        for epoch in tqdm(range(config["epochs"]), desc="Epochs"):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
            val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            })

            tqdm.write(f"Epoch {epoch+1}/{config['epochs']} | "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Gradients 
        biases = [p.grad.detach().cpu() for p in model.parameters() if p.grad is not None and p.dim() == 1]
        weights = [p.grad.detach().cpu() for p in model.parameters() if p.grad is not None and p.dim() == 2]

        bias_norms = [(p * p).sum().sqrt().item() for p in biases]
        weight_norms = [(p * p).sum().sqrt().item() for p in weights]

        plt.figure()
        plt.plot(bias_norms, marker="o")
        plt.plot(weight_norms, marker="s")
        plt.xlabel("Layer")
        plt.ylabel(r"$||\nabla_p \mathcal{L}||$")
        plt.legend(["Biases", "Weight Matrices"])
        plt.title("Gradient Norms " + config["run_name"] + " Epoch " + str(epoch+1))
        plt.tight_layout()
            
        img_filename = "output/grad_norms" + config["run_name"] + "_depth_" + str(depth) + ".png"
        plt.savefig(img_filename)
        plt.close()

        # Log to wandb
        wandb.log({"gradient_norms": wandb.Image(img_filename)})
        
        #torch.save(model.state_dict(), os.path.join(wandb.run.dir, "ResCNN_CAM.pth"))
        
        run.finish()
    return model


# Config for each run

if __name__ == "__main__":
    
    # MLP
    
    # config = {
    #     "project_name": "Lab1_experiments_MNIST",
    #     "data": "MNIST",
    #     "run_name": "MLP_baseline",
    #     "model": "MLP", 
    #     "input_size": 28*28,
    #     "hidden_layers": [128, 64, 128],
    #     "depths" : [1, 5, 10, 15],
    #     "output_size" : 10,
    #     "batch_size": 64,
    #     "lr": 0.001,
    #     "epochs": 10,
    # }
    
    # ResMLP
    
    # config = {
    #     "project_name": "Lab1_experiments_MNIST",
    #     "data": "MNIST",
    #     "run_name": "ResMLP",
    #     "model": "ResMLP",
    #     "input_size": 28*28,
    #     "hidden_layers": [128, 64, 128],
    #     "depths" : [1, 5, 10, 15],
    #     "output_size" : 10,
    #     "batch_size": 64,
    #     "lr": 0.001,
    #     "epochs": 10,
    # }
    
    # CNN
    
    # config = {
    #     "project_name": "Lab1_experiments_CIFAR10",
    #     "data": "CIFAR10",
    #     "run_name": "PlainCNN_baseline",
    #     "model": "PlainCNN",
    #     "input_size": 3, #Note for CNN Refered to number of channel in input
    #     #"hidden_layers" not needed 
    #     "depths": [1, 5, 10, 15],
    #     "output_size" : 10,
    #     "batch_size": 64,
    #     "lr": 0.001,
    #     "epochs": 20,
    # }

    #ResCNN
    
    config = {
        "project_name": "Lab1_experiments_CIFAR10", #"ResCNN_CAM",
        "data": "CIFAR10",
        "run_name": "ResCNN",
        "model": "ResCNN",
        "input_size": 3, #Note for CNN Refered to number of channel in input
        #"hidden_layers" not needed 
        "depths": [1, 5, 10, 15],
        "output_size" : 10,
        "batch_size": 64,
        "lr": 0.001,
        "epochs": 20,
    }

    train_model(config)
