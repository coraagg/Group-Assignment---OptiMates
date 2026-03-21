# Group-Assignment--OptiMates
CDS525 CIFAR-100 Image Classification
本项目使用 CIFAR-100 数据集进行图像分类，实现并对比线性模型、MLP、基础 CNN 和优化 CNN（带残差连接）的性能。

## 数据集
- CIFAR-100: 50,000 张训练图像，10,000 张测试图像，100 个类别，每张图像 32×32 彩色 RGB。

## 项目结构
- `utils.py` – 数据加载与预处理（标准化、数据增强）
- `models.py` – 所有模型定义
- `train.py` – 通用训练脚本（支持命令行参数）
- `train_cifar100.ipynb` – Google Colab 笔记本模板
- `logs/` – 训练日志和保存的模型（自动生成）

## 在colab上运行，运行前确认已选择 GPU 运行
https://colab.research.google.com/github/coraagg/Group-Assignment---OptiMates/blob/main/train_cifar100.ipynb

## 依赖
```bash
pip install torch torchvision numpy matplotlib
