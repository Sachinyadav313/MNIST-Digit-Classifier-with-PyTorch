import torch
from data.data import get_mnist_dataloaders
from models.mnist_model import MNISTCNN

train_loader, test_loader = get_mnist_dataloaders()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MNISTCNN().to(device)
model.load_state_dict(torch.load('outputs/checkpoints/mnist_cnn.pth'))
model.eval()

correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total:.2f}%')
