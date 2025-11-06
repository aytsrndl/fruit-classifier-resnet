# 🍎 Fruit Image Classification using ResNet50 (Transfer Learning)

This project builds upon my previous work developing image classification models during my Machine Learning internship at Proya Technology, where I explored multi-label image classification using Google Vertex AI.  
Here, I independently designed and implemented a transfer-learning pipeline using the ResNet50 architecture in TensorFlow/Keras to perform **multi-class classification** on the Fruits-360 dataset.  

By leveraging ResNet’s pretrained ImageNet weights, the model achieves 99.1 % validation accuracy in just 3 epochs, demonstrating the power of transfer learning for small, specialized datasets.  
This repository showcases a fully reproducible deep-learning workflow — from preprocessing and augmentation to model training, evaluation, and visualization — ready to be deployed or extended.

---

## 🚀 Overview

| Component | Description |
|:--|:--|
| **Framework** | TensorFlow 2 / Keras |
| **Model** | ResNet50 (pretrained on ImageNet) + custom Dense head |
| **Dataset** | Fruits-360 (Kaggle) |
| **Task** | Multi-class fruit image classification |
| **Accuracy** | ~99 % validation accuracy in 3 epochs |
| **Hardware** | NVIDIA T4 GPU (Colab) |

---

## 🧠 Model Architecture

| Layer (type) | Output Shape | Parameters |
|:--|:--:|--:|
| ResNet50 (base, frozen) | (None, 7, 7, 2048) | 23 , 587 , 712 |
| GlobalAveragePooling2D | (None, 2048) | 0 |
| Dense (256, ReLU) | (None, 256) | 524 , 544 |
| Dropout (0.3) | (None, 256) | 0 |
| Dense (Softmax, `num_classes`) | (None, 131) | 33 , 667 |
| **Total params** | ≈ 24.1 M (trainable = 0.6 M) | **≈ 97 MB** |

---

## 📊 Training Results

| Metric | Value |
|:--|:--|
| Validation Accuracy | **0.991** |
| Validation Loss | **0.032** |
| Epochs Trained | 3 |
| Batch Size | 8 |
| Image Size | 160 × 160 × 3 |
