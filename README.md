# 🧠 MNIST Digit Classifier with PyTorch [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A convolutional neural network (CNN) that achieves **98.96% accuracy** on the MNIST test set. Designed for educational purposes and easy extension.

![MNIST Samples](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png)*Example digits from MNIST dataset*

## ✨ Features
- **CNN Architecture**: 2 convolutional layers + 2 fully connected layers
- **High Accuracy**: 98.96% on test set
- **Pre-trained Model**: `outputs/checkpoints/mnist_cnn.pth`
- **Training Scripts**: Batch training with progress tracking
- **Image Prediction**: Run inference on custom digit images

## 🛠️ Installation


Clone repository
git clone https://github.com/Sachinyadav313/MNIST-Digit-Classifier-with-PyTorch
cd digit_classifier

Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate # Linux/macOS

.venv\Scripts\activate # Windows
Install dependencies
pip install -r requirements.txt


## 🚀 Usage
### Training
python scripts/train.py
--epochs 10
--batch_size 64
--lr 0.001


### Evaluation
python scripts/evaluate.py
--model outputs/checkpoints/mnist_cnn.pth
--batch_size 64

Accuracy: 98.96% | Loss: 0.0169


### Prediction
python scripts/predict.py
--model outputs/checkpoints/mnist_cnn.pth
--image path/to/your_digit.png


## 🧩 Model Architecture
class MNISTCNN(nn.Module):
def init(self):
super().init()
self.conv1 = nn.Conv2d(1, 32, 3, 1)
self.conv2 = nn.Conv2d(32, 64, 3, 1)
self.fc1 = nn.Linear(9216, 128)
self.fc2 = nn.Linear(128, 10)


## 📂 Project Structure
digit_classifier/
├── data/ # Dataset loading and preprocessing
│ └── data.py
├── models/ # CNN architecture definition
│ └── mnist_model.py
├── outputs/ # Training artifacts
│ └── checkpoints/
│ └── mnist_cnn.pth
├── scripts/ # Executable workflows
│ ├── train.py
│ ├── evaluate.py
│ └── predict.py
├── requirements.txt # Python dependencies
└── README.md # This document


## 🤝 Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add some amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.

## 🙏 Acknowledgements
- [MNIST Database](http://yann.lecun.com/exdb/mnist/) for the benchmark dataset
- PyTorch team for the deep learning framework
