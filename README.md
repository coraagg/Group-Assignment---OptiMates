# 🖼️ From Linear Baseline to Residual CNN: Image Classification on CIFAR-100  
**Practical Application of Deep Learning (CDS525)**  
**Group Name**: OptiMates  

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org)  
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org)  
[![CIFAR-100](https://img.shields.io/badge/Dataset-CIFAR--100-20B2AA.svg)](https://www.cs.toronto.edu/~kriz/cifar.html)  
[![Colab](https://img.shields.io/badge/Run%20in-Colab-F9AB00.svg)](https://colab.research.google.com/github/coraagg/Group-Assignment---OptiMates/blob/main/notebooks/train_cifar100.ipynb)

---

## 📌 Project Overview

This project systematically explores deep learning for image classification on the **CIFAR-100** dataset — a challenging benchmark with 100 fine-grained classes and only 600 images per class.

We build and compare models of increasing complexity:
- ✅ **Linear Model** – Simple baseline to understand the difficulty of the task.
- ✅ **Multi-Layer Perceptron (MLP)** – With ablation studies on hidden size, activation, dropout, and weight decay.
- ✅ **Basic CNN** – A standard convolutional architecture.
- ✅ **Optimized Residual CNN** – Inspired by ResNet, featuring batch normalization, residual connections, dropout, and global average pooling.

**Final Optimized CNN Test Accuracy**: **56.29%**  
This represents a **+47 percentage point** improvement over the linear baseline (9.64%) and a **+20 point** gain over the basic CNN (36.31%).

---

## 📋 Table of Contents
- [Repository Structure](#repository-structure)
- [Key Results](#key-results)
- [Setup and Installation](#setup-and-installation-reproducibility)
- [How to Run & Reproduce](#how-to-run--reproduce-results)
- [Loading a Trained Model](#loading-a-trained-model)
- [Full Project Report](#full-project-report)
- [Technologies Used](#technologies-used)
- [Team Members](#team-members)

---

## 📁 Repository Structure

```text
Group-Assignment---OptiMates/
├── src/                         # Source code
│   ├── models.py                # All model definitions (Linear, MLP, CNN, ResNet-style)
│   ├── utils.py                 # Data loading, normalization, augmentation
│   ├── train.py                 # Universal training script with argparse
│   └── linear_plot_results.py   # Plotting utility for linear model
├── notebooks/                   # Jupyter/Colab notebooks
│   └── train_cifar100.ipynb     # Main notebook for training & visualization
├── results/                     # All experiment outputs
│   ├── csv/                     # Training logs (loss, accuracy per epoch)
│   ├── figures/                 # Generated plots (learning curves, comparisons, predictions)
│   └── weights/                 # Best model checkpoints (.pth files)
├── requirements.txt
├── LICENSE
└── README.md
```

## 📊 Key Results
Model	Test Accuracy	Key Techniques
Linear Model	9.64%	Flatten + FC
Best MLP	36.20%	2048 hidden units, ReLU, weight decay 5e-4
Basic CNN	36.31%	Two conv layers + FC
Optimized CNN	56.29%	Residual blocks, BN, dropout 0.3, AdamW, data augmentation
Detailed ablation studies (hidden size, activation, regularization) and hyperparameter contrasts (loss functions, learning rates, batch sizes) are available in the report and visualized in results/figures/.

## ⚙️ Setup and Installation (Reproducibility)
### Prerequisites
- Python 3.9 or higher
- Git

### Step-by-step
1. Clone the repository
```
git clone https://github.com/coraagg/Group-Assignment---OptiMates.git
cd Group-Assignment---OptiMates
```
2. Create virtual environment (recommended)
```
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```
3. Install all dependencies
```
pip install -r requirements.txt
```
4. Data
The CIFAR-100 dataset will be automatically downloaded by torchvision when you first run the training script or notebook.
   
## ▶️ How to Run & Reproduce Results
### Option 1: Run the Notebook (Recommended)
- Open the notebook in Google Colab:
https://img.shields.io/badge/Run%2520in-Colab-F9AB00.svg
- Or run locally:
```
jupyter notebook notebooks/train_cifar100.ipynb
```
- Execute cells in order to train models, generate plots, and view sample predictions.

### Option 2: Use the Training Script
- Train a specific model from the command line:
```
# Example: Train the optimized CNN
python src/train.py --model optimized_cnn --epochs 50 --batch_size 128 --lr 0.01

# Example: Train the MLP with custom hidden size
python src/train.py --model mlp --hidden_size 2048 --dropout 0.3 --weight_decay 5e-4
```
- Check src/train.py -h for all available arguments.
All training logs and model weights will be saved to the results/ folder automatically.

## 🔄 Loading a Trained Model
```
import torch
from src.models import OptimizedCNN

# Load model architecture
model = OptimizedCNN(num_classes=100)

# Load saved weights (adjust path to your preferred checkpoint)
checkpoint = torch.load('results/weights/optimized_cnn_best.pth', map_location='cpu')
model.load_state_dict(checkpoint)
model.eval()

print("✅ Model loaded successfully!")
```

## 📘 Full Project Report
The complete group report (including problem definition, methodology, ablation studies, contrast experiments, and analysis) is available in the repository root as:
Group-Assignment---OptiMates.pdf

## 🛠️ Technologies Used
- Python 3.9+
- Deep Learning: PyTorch, torchvision
- Visualization: Matplotlib
- Numerical: NumPy
- Environment: Jupyter Notebook / Google Colab

Thank you for checking out our project!
We hope this structured, fully reproducible repository clearly demonstrates our deep learning pipeline for CIFAR-100 classification.
🚀
