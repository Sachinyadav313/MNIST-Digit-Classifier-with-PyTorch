import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.data import get_mnist_dataloaders
from models.mnist_model import MNISTCNN


train_loader, test_loader = get_mnist_dataloaders()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MNISTCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1, 6):  # 5 epochs
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch}, Loss: {running_loss/len(train_loader):.4f}")
