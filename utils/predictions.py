import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils.image_upload import normalize_channel

ML_INFERENCE_SECONDS = 0.08


def _preprocess_pair(ed_image: np.ndarray, es_image: np.ndarray) -> np.ndarray:
    from skimage.transform import resize

    ed_resized = normalize_channel(resize(ed_image, (128, 128)))
    es_resized = normalize_channel(resize(es_image, (128, 128)))
    return np.stack([ed_resized, es_resized], axis=-1).reshape(1, 128, 128, 2).astype(np.float32)


def predict_with_model(ed_image, es_image, model):
    X = _preprocess_pair(ed_image, es_image)
    pred = model.predict(X, verbose=0)
    ca_pred = float(pred[0, 0])
    cb_pred = float(pred[0, 1])
    return ca_pred, cb_pred, "Real Model"


def predict_with_uncertainty(ed_image, es_image, model, n_samples: int = 30):
    X = _preprocess_pair(ed_image, es_image)
    outputs = np.stack([np.asarray(model(X, training=True))[0] for _ in range(n_samples)])
    ca_mean = float(outputs[:, 0].mean())
    ca_std = float(outputs[:, 0].std())
    cb_mean = float(outputs[:, 1].mean())
    cb_std = float(outputs[:, 1].std())
    return ca_mean, ca_std, cb_mean, cb_std


def compute_real_gradcam(model, ed_image, es_image, output_index: int = 0):
    import tensorflow as tf
    from skimage.transform import resize

    X = _preprocess_pair(ed_image, es_image)

    last_conv_layer = None
    for layer in reversed(model.layers):
        if "conv" in layer.__class__.__name__.lower():
            last_conv_layer = layer
            break
    if last_conv_layer is None:
        return None

    grad_model = tf.keras.Model(inputs=model.inputs, outputs=[last_conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(X)
        target = predictions[:, output_index]

    grads = tape.gradient(target, conv_output)
    if grads is None:
        return None
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    heatmap = heatmap / max_val if max_val > 0 else heatmap
    heatmap = heatmap.numpy()

    return resize(heatmap, (128, 128))


def overall_metrics(df: pd.DataFrame) -> dict:
    metrics = {}
    for param in ("ca", "cb"):
        gt = df[f"{param}_gt"].values
        pred = df[f"{param}_pred"].values
        metrics[param] = {
            "mae": mean_absolute_error(gt, pred),
            "rmse": np.sqrt(mean_squared_error(gt, pred)),
            "r2": r2_score(gt, pred),
        }
    return metrics


def mae_by_pathology(df: pd.DataFrame, param: str = "ca") -> pd.Series:
    return df.groupby("pathology", observed=True).apply(
        lambda g: mean_absolute_error(g[f"{param}_gt"], g[f"{param}_pred"])
    )
