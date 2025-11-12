# Fish Classification Using Deep Learning (TensorFlow & MobileNetV2)

## Overview

* This project focuses on developing an **AI-based Fish Classification System** using **TensorFlow** and **Keras**.  
* The model leverages **transfer learning** with **MobileNetV2**, a lightweight yet powerful CNN architecture pre-trained on ImageNet.  
* It accurately classifies fish species from images, helping automate biodiversity research, fisheries monitoring, and aquaculture systems.

---

##  Features
- Deep Learning–based image classification using **MobileNetV2**
- **Transfer Learning** and **Fine-Tuning** for high accuracy
- **Data Augmentation** for improved model generalization
- **Confusion Matrix** visualization for performance analysis
- **Model Saving and Prediction** on custom images

---

## Architecture
    ┌────────────────────────────┐
    │     Image Dataset          │
    └────────────┬───────────────┘
                 │
     ┌───────────▼────────────┐
     │   Data Augmentation    │
     └───────────┬────────────┘
                 │
     ┌───────────▼────────────┐
     │   Preprocessing Layer  │
     └───────────┬────────────┘
                 │
     ┌───────────▼────────────┐
     │   MobileNetV2 Base     │
     │  (Feature Extraction)  │
     └───────────┬────────────┘
                 │
     ┌───────────▼────────────┐
     │ GlobalAvgPooling + BN  │
     │ Dropout(0.3)           │
     │ Dense(Softmax) Layer   │
     └───────────┬────────────┘
                 │
     ┌───────────▼────────────┐
     │   Prediction Output    │
     └────────────────────────┘

---

## Technologies Used
- **Python 3**
- **TensorFlow / Keras**
- **Matplotlib** – for data visualization  
- **scikit-learn** – for confusion matrix and evaluation metrics  
- **Google Colab** – for model training and testing  
- **PIL (Pillow)** – for image handling  

---

## Dataset
The dataset used is **NA_Fish_Dataset**, consisting of multiple fish species organized into:

---

## ⚙️ Key Files Explained

```
Fish_Classification_Project/
│
├── 📂 NA_Fish_Dataset/
│ ├── 📂 train/ # Training dataset (80%)
│ │ ├── Class_1/ # Example: Tuna
│ │ ├── Class_2/ # Example: Salmon
│ │ ├── Class_3/ # Example: Trout
│ │ └── ...
│ │
│ └── 📂 val/ # Validation dataset (20%)
│ ├── Class_1/
│ ├── Class_2/
│ ├── Class_3/
│ └── ...
│
├── 🧩 fc_model.keras # Saved trained model file
│
├── 📄 Fish_Classification.ipynb # Main notebook/code file
```
---



You can replace this dataset with your own structured image dataset for custom training.

---

## Installation & Setup

### Step 1: Install Required Packages
```python
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import zipfile
import os
```
### Step 2: Upload and Extract Dataset
### Step 3: Load and Prepare Dataset
### Step 4: Build and Train the Model
### Step 5: Fine-Tuning
### Step 6: Evaluate Model
### Step 7: Save and Download the Model
### Step 8: Predict Custom Image

## Results

Achieved high validation accuracy through transfer learning

Model successfully classified test images with strong confidence

Visual results include confusion matrix and accuracy/loss graphs

## SDG Achievement

SDG 14 – Life Below Water: Supports marine biodiversity monitoring

SDG 9 – Innovation & Infrastructure: Encourages AI-driven innovation

SDG 13 – Climate Action: Aids in aquatic ecosystem study

SDG 4 – Quality Education: Promotes AI learning in sustainability

## Conclusion

This project demonstrates how AI and deep learning can enhance marine research through automated fish classification.
The system is scalable, accurate, and applicable for environmental conservation, education, and smart aquaculture solutions.
