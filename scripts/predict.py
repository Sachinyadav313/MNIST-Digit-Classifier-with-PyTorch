import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import sys

# Add parent dir to path so we can import from models/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.mnist_model import MNISTCNN

# ----------- Load Trained Model -------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MNISTCNN().to(device)
model.load_state_dict(torch.load('outputs/checkpoints/mnist_cnn.pth', map_location=device))
model.eval()

# ----------- Image Preprocessing Function -------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()

# ----------- User Input -------------
if __name__ == "__main__":
    image_path = input("Enter the image path: ").strip()
    if not os.path.exists(image_path):
        print("File not found.")
    else:
        prediction = predict_image(image_path)
        print(f"Predicted Digit: {prediction}")

