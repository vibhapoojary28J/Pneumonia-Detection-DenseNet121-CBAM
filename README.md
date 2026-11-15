# Pneumonia Detection from Chest X-Ray Images Using DenseNet121 + CBAM

A Deep Learning-based Pneumonia Detection System using an Attention-Enhanced DenseNet121 architecture, Transfer Learning, Fine-Tuning and Grad-CAM Explainability.

## 1. Project Overview

Pneumonia is a respiratory infection that affects the lungs and can cause inflammation in the air sacs. Early identification of pneumonia is important for timely medical evaluation and treatment.

This project presents a Deep Learning-based approach for detecting pneumonia from chest X-ray images using an attention-enhanced DenseNet121 architecture.

The proposed model integrates the Convolutional Block Attention Module (CBAM) with DenseNet121 to enhance feature representation through channel and spatial attention mechanisms.

The system classifies chest X-ray images into two categories:

- NORMAL
- PNEUMONIA

Grad-CAM is also implemented to visualize image regions that contribute to the model's predictions.

## 2. Problem Statement

Manual interpretation of chest X-ray images requires medical expertise and can be time-consuming.

The objective of this project is to develop a deep learning model that automatically analyzes chest X-ray images and classifies them as NORMAL or PNEUMONIA.

The project also incorporates attention mechanisms and visual explainability to better understand the model's predictions.

## 3. Project Objectives

- Develop a deep learning model for pneumonia detection.
- Implement DenseNet121 using ImageNet pretrained weights.
- Integrate CBAM attention mechanisms.
- Apply transfer learning and fine-tuning.
- Handle class imbalance using class weights.
- Evaluate the model using multiple performance metrics.
- Implement Grad-CAM for visual explainability.
- Develop reusable prediction utilities for future application integration.

## 4. Dataset

**Dataset:** Chest X-Ray Images (Pneumonia)

**Source:** Kaggle

Dataset Link:
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

The dataset contains chest X-ray images belonging to two categories:

- NORMAL
- PNEUMONIA

The dataset is downloaded programmatically using KaggleHub.

The original training and validation data are reorganized into a stratified development split, while the original test dataset is kept separate for final evaluation.

## 5. Proposed Architecture

### DenseNet121 + CBAM Attention

The proposed architecture combines DenseNet121 with the Convolutional Block Attention Module (CBAM).

**DenseNet121:**
- Uses dense connections between layers.
- Encourages feature reuse.
- Supports effective gradient propagation.
- Uses ImageNet pretrained weights for transfer learning.

**CBAM Attention Module:**

CBAM consists of two attention mechanisms:

1. Channel Attention: Emphasizes informative feature channels.
2. Spatial Attention: Highlights important spatial regions in feature maps.

### Model Architecture

```text
Input Chest X-Ray Image
         |
         v
 Image Preprocessing
         |
         v
    DenseNet121
   Feature Extraction
         |
         v
 Channel Attention
         |
         v
 Spatial Attention
         |
         v
 Global Average Pooling
         |
         v
       Dropout
         |
         v
 Classification Layer
         |
         v
   Sigmoid Activation
         |
         v
   Binary Prediction
         |
    -------------
    |           |
  NORMAL    PNEUMONIA
```

## 6. Methodology

The project follows these stages:

1. Dataset Acquisition
2. Exploratory Data Analysis
3. Image Preprocessing
4. Data Augmentation
5. Class Imbalance Handling
6. DenseNet121 Feature Extraction
7. CBAM Attention Integration
8. Transfer Learning
9. Fine-Tuning
10. Model Evaluation
11. Grad-CAM Visualization
12. Model Saving and Prediction

### Training Strategy

The model is trained in two stages.

**Stage 1: Transfer Learning**

The pretrained DenseNet121 backbone is frozen, and the newly added attention and classification layers are trained.

**Stage 2: Fine-Tuning**

Selected upper layers of DenseNet121 are unfrozen and trained with a smaller learning rate to adapt the pretrained features to the chest X-ray dataset.

Training callbacks include:

- EarlyStopping
- ModelCheckpoint
- ReduceLROnPlateau

## 7. Model Evaluation

The final model is evaluated using the untouched test dataset.

The evaluation metrics include:

- Accuracy
- Precision
- Recall (Sensitivity)
- Specificity
- F1-Score
- ROC-AUC

The project also generates:

- Confusion Matrix
- Classification Report
- ROC Curve
- Precision-Recall Curve
- Training Accuracy Graph
- Training Loss Graph

Actual evaluation values are available in the `results/` directory.

## 8. Explainability Using Grad-CAM

Grad-CAM (Gradient-weighted Class Activation Mapping) is implemented to visualize the image regions contributing to the model's predictions.

It generates heatmaps and overlays them on the original chest X-ray images.

This provides a visual interpretation of the model's decision-making process.

Grad-CAM is used as an explainability technique and does not establish clinical validity.

## 9. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming |
| TensorFlow / Keras | Deep Learning |
| DenseNet121 | Feature Extraction |
| CBAM | Attention Mechanism |
| NumPy | Numerical Operations |
| Pandas | Data Analysis |
| Matplotlib | Visualization |
| Seaborn | Statistical Visualization |
| Scikit-learn | Evaluation Metrics |
| KaggleHub | Dataset Download |
| Jupyter Notebook | Model Development |
| VS Code | Development Environment |

## 10. Project Structure

```text
Pneumonia-Detection-DenseNet121-CBAM/
│
├── dataset/
│
├── notebooks/
│   └── Pneumonia_Detection_DenseNet121_CBAM.ipynb
│
├── saved_models/
│   └── best_pneumonia_model.keras
│
├── results/
│   ├── training_accuracy.png
│   ├── training_loss.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   ├── classification_report.txt
│   ├── evaluation_metrics.json
│   └── gradcam/
│
├── sample_images/
│
├── inference_utils.py
│
├── requirements.txt
│
├── README.md
│
└── .gitignore
```

Note: The dataset is excluded from the public GitHub repository.

## 11. Installation and Setup

### Prerequisites

- Python
- Visual Studio Code
- Jupyter Extension
- Required Python dependencies
- NVIDIA GPU (optional)

### Step 1: Clone the Repository

```bash
git clone https://github.com/vibhapoojary28J/Pneumonia-Detection-DenseNet121-CBAM.git
```

### Step 2: Navigate to the Project

```bash
cd Pneumonia-Detection-DenseNet121-CBAM
```

### Step 3: Create a Virtual Environment

```bash
python -m venv pneumonia_env
```

### Step 4: Activate the Environment

Windows:

```bash
pneumonia_env\Scripts\activate
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Open the Notebook

Open the `notebooks/` folder in VS Code and execute:

`Pneumonia_Detection_DenseNet121_CBAM.ipynb`

Follow the notebook cells sequentially.

Kaggle authentication may be required for downloading the dataset through KaggleHub.

## 12. Model Inference

The project includes `inference_utils.py`, which provides reusable utilities for loading the trained model and predicting individual chest X-ray images.

The saved model can be loaded for inference and integrated into a separate application.

The model is saved in Keras format:

```text
saved_models/best_pneumonia_model.keras
```

## 13. Future Enhancements

- Develop an interactive Streamlit-based user interface.
- Enable chest X-ray image uploads.
- Display prediction results and confidence scores.
- Integrate Grad-CAM visualizations into the interface.
- Evaluate the model using additional external datasets.
- Perform further model optimization and validation.

## 14. Limitations

- The model is trained using a single publicly available dataset.
- Dataset distribution may not represent all patient populations.
- External clinical validation has not been performed.
- Grad-CAM visualizations are approximate explanations and may not identify medically relevant regions.
- The model is an academic research prototype and is not intended for clinical diagnosis.

## 15. Conclusion

This project explores an attention-enhanced DenseNet121 architecture for automated pneumonia classification from chest X-ray images.

By combining transfer learning, CBAM attention mechanisms, fine-tuning and Grad-CAM visualization, the project aims to develop an interpretable deep learning-based classification pipeline.

The implementation provides a foundation for further research and future integration into an interactive application.

---

**Project Category:** Neural Networks and Deep Learning (NNDL)

**Project Type:** B.Tech AIML Open-Ended Project

**Model:** DenseNet121 + CBAM Attention

**Classification:** NORMAL / PNEUMONIA
