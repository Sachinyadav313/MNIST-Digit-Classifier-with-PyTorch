# 🧠 MNIST Digit Classifier with PyTorch

A CNN-based digit classifier built using PyTorch. This project trains a model on the MNIST dataset and provides scripts for training, evaluation, and predicting digits from images.

---

## 📁 Project Structure

digit_classifier/
├── data/
│ └── data.py # Loads MNIST dataset
├── models/
│ └── mnist_model.py # CNN model definition
├── outputs/
│ └── checkpoints/
│ └── mnist_cnn.pth # Saved trained model
├── scripts/
│ ├── train.py # Training script
│ ├── evaluate.py # Evaluation script
│ └── predict.py # Predict digit from image
├── .venv/ # Virtual environment (recommended)
├── requirements.txt # Required packages
└── README.md # Project overview

python scripts/evaluate.py
Accuracy: 98.96%
