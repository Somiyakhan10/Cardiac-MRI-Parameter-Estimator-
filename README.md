# MyoParam Estimator

A Streamlit application that estimates patient-specific Holzapfel-Ogden (HO) passive myocardial material parameters - two scaling factors, Ca and Cb - from paired end-diastolic (ED) and end-systolic (ES) cardiac MRI slices using a trained convolutional neural network, replacing hours of iterative inverse finite-element optimization with sub-second inference.

---
## Output:
Overview:
<img width="1828" height="811" alt="image" src="https://github.com/user-attachments/assets/2f300555-99fb-413a-97a9-400b4e8aab68" />
Methodology:
<img width="1849" height="787" alt="image" src="https://github.com/user-attachments/assets/23a044c4-c456-46d0-a119-9b78df6d371c" />
Interactive Demo:
<img width="1482" height="842" alt="image" src="https://github.com/user-attachments/assets/4df7e96a-8ce5-4e2d-a856-290c20d24a63" />
Model Performance:
<img width="1501" height="839" alt="image" src="https://github.com/user-attachments/assets/df74b5e4-3bb0-4794-84d9-276f5aeab63e" />
Clinical Translation
<img width="1482" height="610" alt="image" src="https://github.com/user-attachments/assets/783b3b9f-c5fb-46c5-b2dd-d05ff381e7be" />

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

## Usage

1. Open the Interactive Demo tab
2. Upload an End-Diastolic (ED) image
3. Upload an End-Systolic (ES) image
4. For 3D volumes, select the appropriate slice
5. For 4D volumes, select both the cardiac phase and the slice
6. View predicted Ca and Cb values with uncertainty estimates
7. Download prediction report as plain text
8. View Grad-CAM attention map in the Model Performance tab

---

## Preprocessing

No training-time preprocessing script is available for the supplied model. Each ED and ES channel is:

- Percentile-clipped (1st and 99th percentile)
- Scaled to the range [0, 1] independently
- Resized to 128x128 pixels
- Stacked into the model's input tensor (128x128x2)

This is a documented assumption rather than a verified match to the original training pipeline.

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

## Uncertainty Estimation

Predictions are reported as mean and standard deviation obtained via Monte Carlo Dropout (Gal and Ghahramani, 2016). The model's existing dropout layers remain active at inference time across 30 stochastic forward passes on the same input.



## License

This project is for research and demonstration purposes only. Not intended for clinical use without proper validation.



