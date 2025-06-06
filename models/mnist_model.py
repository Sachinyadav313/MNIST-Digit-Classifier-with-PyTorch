import torch
import torch.nn as nn 
import torch.nn.functional as F

class MNISTCNN(nn.Module):
    def __init__(self):
        super(MNISTCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # 32x28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 64x28x28
        self.pool = nn.MaxPool2d(2, 2)                            # After pool: 64x14x14 → 64x7x7
        
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # → 32x14x14
        x = self.pool(F.relu(self.conv2(x)))  # → 64x7x7
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
