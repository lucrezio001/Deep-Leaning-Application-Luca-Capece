import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock
import torch.nn.functional as F

# MLP

class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers, depth, output_size):
        super().__init__()
        layers = [nn.Flatten()]
        current_size = input_size
        for _ in range(depth):
            for hidden_size in hidden_layers:
                layers.append(nn.Linear(current_size, hidden_size))
                layers.append(nn.ReLU())
                current_size = hidden_size

        layers.append(nn.Linear(current_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ResMPL

class ResMLPBlock(nn.Module):
    def __init__(self, hidden_layers):
        super().__init__()
        layers = []
        input_dim = hidden_layers[0]
        
        for i in range(1, len(hidden_layers)):
            hidden_size = hidden_layers[i]
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            input_dim = hidden_size
        
        self.block = nn.Sequential(*layers)
        
    def forward(self, x):
        out = self.block(x)
        return F.relu(out + x)
        
class ResMLP(nn.Module):
    def __init__(self, input_size, hidden_layers, depth, output_size):
        super().__init__()
        self.input = nn.Linear(input_size, hidden_layers[0])
        self.blocks = nn.Sequential(*([ResMLPBlock(hidden_layers)] * depth))
        self.output= nn.Linear(hidden_layers[-1], output_size)
    
    def forward(self, x):
        x = x.flatten(1)
        x = F.relu(self.input(x))
        x = self.blocks(x)
        return self.output(x)

# CNN

# Copy of resblock without residual connection
class PlainCNNBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        return out

class PlainCNN(nn.Module):
    def __init__(self, depth, in_channels = 3, output_size=10):
        super().__init__()
        layers = []
        
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        for _ in range(depth):
            layers.append(PlainCNNBlock(64, 64))

        self.cnn_blocks = nn.Sequential(*layers)
        
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64, output_size)

    def forward(self, x):
        x = self.adapter(x)
        x = self.cnn_blocks(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# ResCNN

class ResCNN(nn.Module):
    def __init__(self, depth,in_channels = 3, output_size=10):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        blocks = []
        for _ in range(depth):
            blocks.append(BasicBlock(64, 64)) # Basic block taken from resnet18
        self.res_cnn_block = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64, output_size)

    def forward(self, x):
        x = self.adapter(x)
        x = self.res_cnn_block(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)