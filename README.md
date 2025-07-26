# ASD Prediction in Children Using Deep Learning

**Early Autism Spectrum Disorder (ASD) Detection via Facial Image Analysis**

## 📌 Project Overview

This project aims to predict Autism Spectrum Disorder (ASD) in children using facial images and deep learning models. The goal is to provide an early, non-invasive diagnostic aid by leveraging the feature extraction capabilities of CNN-based architectures.

## 🧠 Models Used

1. **VGG19**
2. **ResNet50V2**
3. **InceptionV3**
4. **Hybrid Model (InceptionV3 + ResNet50V2)**

All models were fine-tuned using transfer learning with ImageNet weights and custom classification heads for binary classification (ASD vs. Non-ASD).

## 📂 Dataset

- Facial image dataset containing labeled images:
  - `autism/` and `non-autism/` classes
  - Preprocessed with resizing and normalization
- Augmentation techniques applied to improve generalization

## ⚙️ Implementation Details

- Developed in **Google Colab**
- Used Keras & TensorFlow with GPU acceleration
- Layers:
  - Frozen pretrained layers
  - Added custom Dense and Dropout layers
- Binary cross-entropy loss with Adam optimizer

## 📊 Evaluation Metrics

- Accuracy
- Precision, Recall
- Confusion Matrix
- Classification Report

### ✅ Sample Results:

| Model             | Accuracy |
|------------------|----------|
| VGG19            | ~91%     |
| ResNet50V2       | ~93%     |
| InceptionV3      | ~92%     |
| Hybrid Model     | **~95%** |

## 📈 Visualizations

- Training/Validation Loss & Accuracy plots
- Confusion matrix heatmaps
- Model comparison bar charts

## 🛠 Technologies Used

- Python, Google Colab
- TensorFlow, Keras
- NumPy, OpenCV
- Matplotlib, Seaborn



