# MyoParam Estimator

A Streamlit application that estimates patient-specific Holzapfel-Ogden (HO) passive
myocardial material parameters — two scaling factors, Ca and Cb — from a paired
end-diastolic (ED) and end-systolic (ES) cardiac MRI slice, using a trained
convolutional neural network in place of iterative inverse finite-element
optimization.

## Scope

Predictions are produced only from user-uploaded ED/ES images and the weights of a
trained Keras model. There is no bundled dataset and no synthetic fallback. The
model's original training configuration and the Ca/Cb labels it was trained against
are not documented in this project; predictions should be treated as a demonstration
of the estimation pipeline rather than a clinically validated measurement. No
ground-truth value is available for uploaded images, so no accuracy claim is made
about any individual prediction — instead, each prediction is reported as a mean and
standard deviation obtained via Monte Carlo Dropout (Gal and Ghahramani, 2016), using
the model's existing dropout layers over 30 stochastic forward passes.

## Requirements

- Python 3.11
- A trained Keras model file matching a 128×128×2 input (stacked ED and ES channels)
  and a 2-value output (Ca, Cb)

## Setup

```bash
pip install -r requirements.txt
```

Place the trained model at the path defined by `config.MODEL_PATH`
(`models/acdc_parameter_estimator.keras` by default). If the file is absent or fails
to load, the application still starts; the Interactive Demo and Model Performance
tabs report that no prediction is possible and state why.

```bash
streamlit run app.py
```

## Usage

1. Open the Interactive Demo tab.
2. Upload an end-diastolic (ED) image and an end-systolic (ES) image. Both are
   required — a single image, or the same image supplied twice, provides no motion
   information between cardiac phases and is not accepted.
3. For a 3D volume, select the slice; for a 4D volume (a full cine sequence), select
   both the cardiac phase and the slice.
4. Review the predicted Ca and Cb values, their uncertainty, and the inference-speed
   comparison. A plain-text prediction report can be downloaded.
5. In the Model Performance tab, view a gradient-based Grad-CAM attention map for the
   uploaded case.

Supported formats: PNG, JPEG, TIFF, NIfTI (`.nii`, `.nii.gz`).

## Preprocessing

No training-time preprocessing script is available for the supplied model. Each of
the ED and ES channels is percentile-clipped (1st/99th percentile) and scaled to
[0, 1] independently, then resized to 128×128 before being stacked into the model's
input tensor. If the original training pipeline used a different normalization, this
is a documented assumption rather than a verified match.

## Project Structure

```
app.py                        Streamlit application
config.py                     Model path configuration
requirements.txt
models/
    acdc_parameter_estimator.keras   Trained model (128x128x2 input, 2-value output)
utils/
    data_loader.py             Shared pathology color/label constants
    image_upload.py            Image loading, normalization, and input validation
    predictions.py             Model inference, Monte Carlo Dropout, Grad-CAM
    visualization.py           Plotly chart construction
assets/
    style.css                  Application theme
```

## Notes

- Grayscale conversion in `utils/image_upload.py` handles standard 8-bit modes,
  palette images, RGB/RGBA/CMYK (by channel averaging, with alpha discarded), and
  higher bit-depth or floating-point single-channel modes without a lossy 8-bit
  conversion step.
- The application performs two input sanity checks before displaying a prediction:
  a near-constant (low-contrast) image warning, and an identical-ED/ES warning.
  Neither blocks the prediction; both are advisory.
- A model file must be saved with a Keras version compatible with the one installed
  (see `requirements.txt`); a version mismatch raises a deserialization error naming
  the specific incompatible layer or configuration key.
