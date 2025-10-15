# 🧠 Brain Tumor Classification using Deep Learning (InceptionV3 + Grad-CAM)

This project implements a **deep learning solution** for **multi-class classification of brain tumors** from MRI images.  
Leveraging **Transfer Learning** with **InceptionV3**, the model accurately distinguishes between different tumor types and normal brain tissue.  
Using **Grad-CAM**, the system provides interpretable visualizations highlighting regions of the MRI that most influenced each prediction.  
A **Streamlit web app** was developed for interactive online demonstration and inference.

---

## 🚀 Key Features

- 🧩 **Transfer Learning with InceptionV3**
  - Fine-tuned on MRI scans for efficient convergence and high accuracy.
- 📊 **Multi-Class Classification**
  - Detects and classifies multiple types of brain tumors (e.g., *glioma*, *meningioma*, *pituitary*, *no tumor*).
- 🔍 **Explainability with Grad-CAM**
  - Visualizes salient image regions driving model decisions for interpretability.
- 🌐 **Interactive Streamlit App**
  - Simple drag-and-drop interface for real-time model predictions 
- 📈 **Model Performance**
  - Achieved **Recall = 94%** on a held-out test set.

---

## 🧬 Dataset

- **Source:** Publicly available brain MRI datasets (e.g., [Kaggle Brain Tumor Dataset]([[https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)]))  
- **Classes:**  
  - Glioma Tumor  
  - Meningioma Tumor  
  - Pituitary Tumor  
  - No Tumor  
- **Preprocessing:**
  - Image resizing (e.g., 224×224)
  - Normalization to [0, 1]

---

## 🏗️ Model Architecture
Input (224x224x3)
│
├── InceptionV3 (pretrained on ImageNet, frozen base)
├── Conv2D
├── GlobalAveragePooling2D
└── Dense(4, Softmax) → 4 tumor classes


- **Loss Function:** Categorical Cross-Entropy  
- **Optimizer:** Adam  
- **Metrics:** Accuracy, Precision, Recall, F1-Score  

---

## 🧠 Model Performance

| Metric      | Training | Validation | Test |
|--------------|-----------|-------------|------|
| Accuracy     | 99%       | 93%         | 95%  |
| Precision    | --       | --         | 95%  |
| **Recall**   | --   | --    | **95%** |
| F1-Score     | --     | --        | 95%  |

> Achieved **95% recall** on the test set, demonstrating strong sensitivity to tumor detection.

---

## 📊 Grad-CAM Visualization

Grad-CAM (Gradient-weighted Class Activation Mapping) highlights the MRI regions most responsible for the model’s predictions.

| Input MRI | Grad-CAM Heatmap |
|------------|------------------|
| <img width="514" height="390" alt="image" src="https://github.com/user-attachments/assets/d0bd8fa2-8f05-4a11-917e-ac9b9bdf6461" />
| <img width="514" height="390" alt="image" src="https://github.com/user-attachments/assets/c879089e-e090-46de-8108-a2be882f0bec" />
|

---


