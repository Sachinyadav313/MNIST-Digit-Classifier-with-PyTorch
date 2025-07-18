# data/data.py
import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_mnist_dataloaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root='data/', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='data/', train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
