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

| File/Folder | Description |
|--------------|-------------|
| **NA_Fish_Dataset/** | Contains all training and validation images categorized by species. |
| **Fish_Classification.ipynb** | Main Python/TensorFlow notebook containing data loading, model building, training, evaluation, and prediction. |
| **fc_model.keras** | The final saved model file (Keras format) for deployment or future inference. |
| **results/** | Contains generated graphs and metrics like accuracy/loss curves and confusion matrix images. |
| **images/** | Stores supporting visuals like the architecture diagram, SDG icons, and dataset samples. |
| **README.md** | Documentation explaining setup, usage, architecture, and project purpose. |

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
