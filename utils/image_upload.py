import os
import tempfile

import numpy as np

NIFTI_EXTENSIONS = (".nii", ".nii.gz")
STANDARD_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def is_nifti(filename: str) -> bool:
    lower = filename.lower()
    return lower.endswith(NIFTI_EXTENSIONS)


def load_image_file(uploaded_file) -> np.ndarray:
    name = uploaded_file.name.lower()
    if is_nifti(name):
        return _load_nifti(uploaded_file, name)
    return _load_standard_image(uploaded_file)


def _load_nifti(uploaded_file, name: str) -> np.ndarray:
    import nibabel as nib

    suffix = ".nii.gz" if name.endswith(".nii.gz") else ".nii"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name
        img = nib.load(tmp_path)
        data = np.asarray(img.dataobj).astype(np.float32)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    while data.ndim > 2 and data.shape[-1] == 1:
        data = data[..., 0]
    return data


def _load_standard_image(uploaded_file) -> np.ndarray:
    from PIL import Image

    img = Image.open(uploaded_file)
    mode = img.mode

    if mode == "P":
        img = img.convert("RGBA") if "transparency" in img.info else img.convert("RGB")
        mode = img.mode

    if mode == "L":
        return np.array(img).astype(np.float32)

    if mode in ("I", "I;16", "F"):
        return np.array(img).astype(np.float32)

    if mode in ("RGB", "RGBA", "CMYK"):
        arr = np.array(img).astype(np.float32)
        if mode == "RGBA":
            arr = arr[..., :3]
        return arr.mean(axis=-1)

    return np.array(img.convert("L")).astype(np.float32)


def normalize_channel(img2d: np.ndarray, p_low: float = 1, p_high: float = 99) -> np.ndarray:
    lo, hi = np.percentile(img2d, [p_low, p_high])
    if hi <= lo:
        return np.zeros_like(img2d, dtype=np.float32)
    return np.clip((img2d - lo) / (hi - lo), 0, 1).astype(np.float32)


def is_degenerate(img2d: np.ndarray, std_threshold: float = 0.01) -> bool:
    return float(np.std(img2d)) < std_threshold


def images_are_identical(a: np.ndarray, b: np.ndarray, tol: float = 1e-4) -> bool:
    if a.shape != b.shape:
        return False
    return float(np.mean(np.abs(a - b))) < tol
