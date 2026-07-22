# MyoParam Estimator
<div align="center">
    <a href="https://cardiac-mri-parameter-estimator-7mxx.onrender.com/" target="_blank">
        <img src="https://img.shields.io/badge/_Live_Demo-Click_Here-1a237e?style=for-the-badge&logo=render&logoColor=white" alt="Live Demo">
    </a>
</div>
A Streamlit application that estimates patient-specific Holzapfel-Ogden (HO) passive myocardial material parameters - two scaling factors, Ca and Cb - from paired end-diastolic (ED) and end-systolic (ES) cardiac MRI slices using a trained convolutional neural network, replacing hours of iterative inverse finite-element optimization with sub-second inference.

---
## Output

### Dashboard Overview
*The main dashboard interface displaying the project title, sidebar navigation, and the Overview tab with the Holzapfel-Ogden constitutive model explanation and benchmark accuracy metrics.*

<img width="1858" height="705" alt="image" src="https://github.com/user-attachments/assets/00ca3b79-1d6f-41f1-b743-da4368b89322" />

### Methodology
*The Methodology tab illustrating the complete processing pipeline from image upload to parameter estimation, along with network architecture details, parameter grouping visualization, and Monte Carlo Dropout uncertainty quantification method.*

<img width="1498" height="789" alt="image" src="https://github.com/user-attachments/assets/2135efaf-46b4-488e-b096-395396f054fa" />


### Interactive Demonstration
*The Interactive Demo tab featuring dual image uploaders for ED and ES cardiac MRI slices, real-time model inference displaying predicted Ca and Cb values with uncertainty bounds, and a comparative visualization of inference speed against traditional finite element optimization.*

<img width="1502" height="778" alt="image" src="https://github.com/user-attachments/assets/1bd44532-3e3e-4a6d-84a2-3db18e35159f" />


### Model Performance Analysis
*The Model Performance tab presenting benchmark accuracy metrics, Grad-CAM attention maps for model interpretability, and a comprehensive comparison between deep learning predictions and traditional finite element optimization methods.*

<img width="1495" height="792" alt="image" src="https://github.com/user-attachments/assets/cec9456d-ec18-42d8-94b4-2b2bd99e4ff2" />


### Prediction Results
*Example prediction output showing the model's inference on an uploaded cardiac MRI pair, displaying the predicted Ca and Cb material parameters with their uncertainty estimates and the corresponding Grad-CAM attention overlay.*

<img width="1510" height="651" alt="image" src="https://github.com/user-attachments/assets/dcbaa1e2-3d60-445f-9870-ae08b52be393" />


### Clinical Translation
*The Clinical Translation tab outlining potential clinical applications, key benefits, current limitations, and future research directions for myocardial material parameter estimation in cardiovascular disease management.*

<img width="1505" height="611" alt="image" src="https://github.com/user-attachments/assets/79e57041-f6d8-4093-882c-cd2ee5714f92" />


## Overview

Traditional inverse finite-element optimization for myocardial material parameters takes 12 to 24 hours per patient. This application uses a trained CNN to predict Ca (matrix stiffness) and Cb (fiber stiffness) directly from two MRI images in under one second.

| Parameter | Physical Meaning |
|-----------|------------------|
| Ca | Matrix stiffness of myocardial tissue |
| Cb | Fiber stiffness of myocardial tissue |

---

## Features

- Upload ED and ES cardiac MRI images (PNG, JPG, TIFF, NIfTI formats supported)
- Real-time prediction of Ca and Cb values in less than 1 second
- Uncertainty quantification using Monte Carlo Dropout (30 stochastic forward passes)
- Grad-CAM visualization showing model attention maps
- Speed comparison: deep learning (0.08 sec) versus traditional finite element optimization (14.5 hours)
- No dataset required - works with user-uploaded images only

---

## Quick Start

### Prerequisites

- Python 3.11 or higher
- Trained model file (acdc_model_compatible.h5)

### Installation

```bash
git clone https://github.com/Somiyakhan10/MyoParam-Estimator.git
cd MyoParam-Estimator
pip install -r requirements.txt
```

### Place the Model

Place your trained model file in the models directory:

```
models/acdc_model_compatible.h5
```

### Run the Application

```bash
streamlit run app.py
```

---

## Project Structure

```
MyoParam-Estimator/
├── app.py                         Main Streamlit application
├── config.py                      Model path configuration
├── requirements.txt               Python dependencies
├── README.md                      Project documentation
├── models/
│   └── acdc_model_compatible.h5   Trained model (128x128x2 input, 2-value output)
├── utils/
│   ├── image_upload.py            Image loading, normalization, and validation
│   ├── predictions.py             Model inference, Monte Carlo Dropout, Grad-CAM
│   ├── visualization.py           Plotly chart construction
│   └── data_loader.py             Shared constants and helpers
└── assets/
    └── style.css                  Application theme
```
---

## Input Validation

The application performs two sanity checks before displaying a prediction:

- Near-constant (low-contrast) image warning
- Identical ED/ES image warning

Both warnings are advisory and do not block the prediction.

---

## Supported File Formats

- PNG
- JPEG
- TIFF
- NIfTI (.nii, .nii.gz)

---

## License

This project is for research and demonstration purposes only. Not intended for clinical use without proper validation.



