import os
from datetime import datetime

import streamlit as st
import keras
import numpy as np
import pandas as pd

import config
from utils.image_upload import load_image_file, normalize_channel, is_degenerate, images_are_identical
from utils.predictions import predict_with_uncertainty, compute_real_gradcam
from utils.visualization import plot_mri_pair, plot_prediction_bar, plot_speed_comparison, plot_gradcam_overlay

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="MyoParam Estimator",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css():
    css_path = os.path.join(BASE_DIR, "assets", "style.css")
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def pill(value: str, label: str) -> str:
    return f"""<div class="mp-pill">
        <div class="mp-pill-value">{value}</div>
        <div class="mp-pill-label">{label}</div>
    </div>"""


def equal_card(content, height="auto"):
    """Create equal-height cards with consistent styling."""
    return f"""<div class="mp-card" style="min-height:{height}; overflow-y:auto; box-sizing:border-box;">
        {content}
    </div>"""


@st.cache_resource(show_spinner="Loading trained model...")
def load_model():
    return keras.models.load_model(config.MODEL_PATH, compile=False)


def prepare_uploaded_slice(uploaded_file, label: str):
    raw = load_image_file(uploaded_file)
    if raw.ndim == 4:
        frame_idx = st.slider(f"{label}: cardiac phase", 0, raw.shape[3] - 1, 0, key=f"frame_{label}")
        slice_idx = st.slider(f"{label}: slice", 0, raw.shape[2] - 1, raw.shape[2] // 2, key=f"slice_{label}")
        slice2d = raw[:, :, slice_idx, frame_idx]
    elif raw.ndim == 3:
        slice_idx = st.slider(f"{label}: slice", 0, raw.shape[2] - 1, raw.shape[2] // 2, key=f"slice_{label}")
        slice2d = raw[:, :, slice_idx]
    elif raw.ndim == 2:
        slice2d = raw
    else:
        st.error(f"{label}: unsupported image shape {raw.shape}.")
        return None
    return normalize_channel(slice2d)


load_css()

st.markdown(
    """<div class="mp-header">
        <h1>MyoParam Estimator</h1>
        <p>Deep Learning for Myocardial Material Parameter Estimation from Cardiac MRI</p>
    </div>""",
    unsafe_allow_html=True,
)

MODEL = None
MODEL_ERROR = None
if os.path.exists(config.MODEL_PATH):
    try:
        MODEL = load_model()
    except Exception as e:
        MODEL_ERROR = str(e)
else:
    MODEL_ERROR = f"Model file not found at {config.MODEL_PATH}"

USE_REAL_MODEL = MODEL is not None

with st.sidebar:
    st.markdown("### MyoParam Estimator")
    st.caption("AI for Cardiac Tissue Properties")
    st.markdown("---")

    if USE_REAL_MODEL:
        st.success("Trained model loaded")
        st.caption(f"Parameters: {MODEL.count_params():,} · Input: {MODEL.input_shape}")
    else:
        st.error("Model unavailable — predictions cannot be produced this session")
        if MODEL_ERROR:
            with st.expander("Details"):
                st.code(MODEL_ERROR, language=None)

    st.markdown("---")
    st.markdown(
        "**Usage**\n\n"
        "1. Open the Interactive Demo tab\n"
        "2. Upload an end-diastolic (ED) image\n"
        "3. Upload an end-systolic (ES) image\n"
        "4. Review the predicted Ca/Cb parameters"
    )
    st.caption("Supported formats: PNG, JPG, TIFF, NIfTI (.nii, .nii.gz)")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Methodology",
    "Interactive Demo",
    "Model Performance",
    "Clinical Translation",
])

# =========================================================== TAB 1 =========
with tab1:
    st.markdown('<p class="mp-section-title">Project Purpose</p>', unsafe_allow_html=True)
    st.markdown(
        equal_card("""
        This dashboard demonstrates estimation of patient-specific
        <b>Holzapfel-Ogden (HO)</b> passive myocardial material parameters from paired
        end-diastolic (ED) and end-systolic (ES) short-axis cardiac MRI slices using a
        trained convolutional neural network, in place of iterative inverse
        finite-element (FE) optimization. Upload an ED/ES image pair in the Interactive
        Demo tab to run inference against the loaded model.
        """, "120px"),
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="mp-section-title">Holzapfel-Ogden Constitutive Model</p>', unsafe_allow_html=True)
        st.markdown(
            equal_card("""
            <p>The HO model is a structurally-based strain-energy framework describing the
            anisotropic, hyperelastic passive behavior of ventricular myocardium, using
            eight material parameters tied to the fiber (f), sheet (s), and fiber-sheet
            (fs) directions of the tissue microstructure.</p>
            <p>Rather than fitting all eight independently, this project groups them into two
            scaling factors, <b>Ca</b> and <b>Cb</b>, applied to a reference parameter set,
            reducing the estimation problem to a two-value regression learnable directly
            from images.</p>
            """, "340px"),
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown('<p class="mp-section-title">Strain Energy Function</p>', unsafe_allow_html=True)
        with st.container(height=380, border=True):
            st.latex(r"""
            \begin{aligned}
            \psi &= \frac{a}{2b}\exp[b(I_1-3)] \\
            &+ \sum_{i=f,s}\frac{a_i}{2b_i}\left\{\exp\left[b_i(I_{4i}-1)^2\right]-1\right\} \\
            &+ \frac{a_{fs}}{2b_{fs}}\left[\exp\left(b_{fs}I_{8fs}^2\right)-1\right]
            \end{aligned}
            """)
            st.markdown(
                "- **a, b** — isotropic matrix stiffness terms\n"
                "- **a_f, b_f / a_s, b_s** — fiber / sheet direction stiffness\n"
                "- **a_fs, b_fs** — fiber-sheet shear coupling\n"
                "- **I₁, I₄f, I₄s, I₈fs** — deformation invariants\n"
                "- **Ca, Cb** — learned scaling factors"
            )

    st.markdown('<p class="mp-section-title">Motivation</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(pill("12–24 hrs", "Inverse FE optimization per case"), unsafe_allow_html=True)
    with c2:
        st.markdown(pill("< 1 sec", "Direct estimation from images"), unsafe_allow_html=True)
    with c3:
        st.markdown(pill("~50,000×", "Approximate speed-up"), unsafe_allow_html=True)

    st.markdown('<p class="mp-section-title">Benchmark Accuracy</p>', unsafe_allow_html=True)
    results_table = pd.DataFrame({
        "Metric": ["Mean Absolute Error (Ca)", "Mean Absolute Error (Cb)",
                   "Mean Relative Error", "Training Cohort Size",
                   "Input Format", "Reported Inference Time"],
        "Value": ["0.0146", "0.0139", "< 5.00%", "1,288 subjects",
                  "Paired ES + ED mid-ventricular slices", "< 1 second"],
    })
    st.table(results_table.set_index("Metric"))

# =========================================================== TAB 2 =========
with tab2:
    st.markdown('<p class="mp-section-title">Processing Pipeline</p>', unsafe_allow_html=True)
    steps_data = [
        ("1", "Upload", "ED and ES cardiac MRI images"),
        ("2", "Preprocess", "Grayscale, percentile-normalized, resized to 128×128"),
        ("3", "Stack", "ED and ES combined into a 128×128×2 input tensor"),
        ("4", "Inference", "Forward pass through the loaded network"),
        ("5", "Output", "Ca and Cb with Monte Carlo Dropout uncertainty"),
    ]
    cols = st.columns(len(steps_data))
    for col, (num, title, desc) in zip(cols, steps_data):
        with col:
            st.markdown(
                f"""<div class="mp-card" style="height:170px; text-align:center; display:flex; flex-direction:column; justify-content:center; overflow-y:auto;">
                <div style="font-size:1.8rem; font-weight:700; color:var(--mp-secondary);">{num}</div>
                <b>{title}</b><br>
                <span style="font-size:0.85rem;color:#52514e;">{desc}</span>
                </div>""",
                unsafe_allow_html=True,
            )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="mp-section-title">Network Architecture</p>', unsafe_allow_html=True)
        if USE_REAL_MODEL:
            st.markdown(
                equal_card(f"""
                <b>Loaded model</b><br>
                Input shape: {MODEL.input_shape}<br>
                Output shape: {MODEL.output_shape}<br>
                Layers: {len(MODEL.layers)}<br>
                Parameters: {MODEL.count_params():,}<br><br>
                Training configuration (epochs, dataset split, loss curve, and the Ca/Cb
                labels used during training) is not recorded in the model artifact.
                """, "280px"),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                equal_card("""
                Model not currently loaded. Expected input: 128×128×2 (ED and ES
                channels). Expected output: two values (Ca, Cb).
                """, "280px"),
                unsafe_allow_html=True,
            )
    with col2:
        st.markdown('<p class="mp-section-title">Parameter Grouping</p>', unsafe_allow_html=True)
        with st.container(height=320, border=True):
            st.latex(r"a_G = C_a \cdot a_{0G} \qquad\qquad b_G = C_b \cdot b_{0G}")
            st.markdown(
                "Each a-type HO parameter is scaled by a single factor Ca and each "
                "b-type parameter by a single factor Cb, relative to a "
                "population-reference parameter set (a₀, b₀). No training-time "
                "preprocessing script is available for this model; each channel is "
                "percentile-clipped (1st/99th) and scaled to [0, 1] independently "
                "before resizing."
            )

    st.markdown('<p class="mp-section-title">Uncertainty Estimation</p>', unsafe_allow_html=True)
    st.markdown(
        """<div class="mp-card" style="min-height:100px;">
        Predicted values are reported as a mean and standard deviation obtained via
        Monte Carlo Dropout (Gal &amp; Ghahramani, 2016): the model's dropout layers
        remain active at inference time across 30 stochastic forward passes on the
        same input, and the resulting distribution is summarized. This uses the
        model's existing dropout layers and requires no retraining.
        </div>""",
        unsafe_allow_html=True,
    )

# =========================================================== TAB 3 =========
ca_mean = cb_mean = ca_std = cb_std = None
ed_final = es_final = None
ed_name = es_name = None

with tab3:
    st.markdown('<p class="mp-section-title">Upload Cardiac MRI Images</p>', unsafe_allow_html=True)

    if not USE_REAL_MODEL:
        st.error("No prediction is possible — the trained model could not be loaded this session.")
    else:
        uc1, uc2 = st.columns(2)
        with uc1:
            ed_file = st.file_uploader(
                "End-diastolic (ED) image", type=["png", "jpg", "jpeg", "tif", "tiff", "nii", "gz"],
                key="ed_upload",
            )
        with uc2:
            es_file = st.file_uploader(
                "End-systolic (ES) image", type=["png", "jpg", "jpeg", "tif", "tiff", "nii", "gz"],
                key="es_upload",
            )

        if not ed_file or not es_file:
            st.info(
                "Both an ED and an ES image are required. The model estimates tissue "
                "stiffness from the difference between end-diastole and end-systole; "
                "a single image (or the same image used twice) provides no motion "
                "information and is not accepted."
            )
        else:
            ed_final = prepare_uploaded_slice(ed_file, "ED")
            es_final = prepare_uploaded_slice(es_file, "ES")
            ed_name, es_name = ed_file.name, es_file.name

            if ed_final is not None and es_final is not None:
                if is_degenerate(ed_final) or is_degenerate(es_final):
                    st.warning("One of the uploaded images appears to be near-uniform (low contrast). Verify the correct file and slice/frame were selected.")
                if images_are_identical(ed_final, es_final):
                    st.warning("The ED and ES images appear identical. Predictions require genuine motion between the two phases.")

                st.plotly_chart(plot_mri_pair(ed_final, es_final, "Uploaded Case"), use_container_width=True)

                with st.spinner("Running inference..."):
                    ca_mean, ca_std, cb_mean, cb_std = predict_with_uncertainty(ed_final, es_final, MODEL, n_samples=30)

                st.markdown('<p class="mp-section-title">Predicted Material Parameters</p>', unsafe_allow_html=True)
                st.caption("Mean ± standard deviation over 30 Monte Carlo Dropout forward passes")
                p1, p2 = st.columns(2)
                with p1:
                    st.markdown(pill(f"{ca_mean:.3f} ± {ca_std:.3f}", "Predicted Ca"), unsafe_allow_html=True)
                with p2:
                    st.markdown(pill(f"{cb_mean:.3f} ± {cb_std:.3f}", "Predicted Cb"), unsafe_allow_html=True)

                st.plotly_chart(
                    plot_prediction_bar(ca_mean, cb_mean, ca_std, cb_std),
                    use_container_width=True,
                )

                report = (
                    f"MyoParam Estimator — Prediction Report\n"
                    f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n"
                    f"ED file: {ed_name}\n"
                    f"ES file: {es_name}\n\n"
                    f"Model parameters: {MODEL.count_params():,}\n"
                    f"Model input shape: {MODEL.input_shape}\n\n"
                    f"Predicted Ca: {ca_mean:.4f} ± {ca_std:.4f}\n"
                    f"Predicted Cb: {cb_mean:.4f} ± {cb_std:.4f}\n"
                    f"Uncertainty method: Monte Carlo Dropout, 30 stochastic forward passes\n\n"
                    f"Note: no ground-truth Ca/Cb exists for this input; this report "
                    f"records the model's output only.\n"
                )
                st.download_button(
                    "Download prediction report",
                    data=report.encode("utf-8"),
                    file_name="myoparam_prediction_report.txt",
                    mime="text/plain",
                )

    st.markdown('<p class="mp-section-title">Inference Speed</p>', unsafe_allow_html=True)
    st.plotly_chart(plot_speed_comparison(), use_container_width=True)

# =========================================================== TAB 4 =========
with tab4:
    st.markdown('<p class="mp-section-title">Benchmark Accuracy</p>', unsafe_allow_html=True)
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    for col, (label, value) in zip(
        (m1, m2, m3, m4, m5, m6),
        [("Ca MAE", "0.0146"), ("Ca RMSE", "0.0213"), ("Ca R²", "0.89"),
         ("Cb MAE", "0.0139"), ("Cb RMSE", "0.0197"), ("Cb R²", "0.91")],
    ):
        with col:
            st.markdown(pill(value, label), unsafe_allow_html=True)
    st.caption("From prior published literature (1,288-subject cohort); not measured from this session.")

    st.markdown('<p class="mp-section-title">Deep Learning vs. Traditional Inverse FE</p>', unsafe_allow_html=True)
    comparison_table = pd.DataFrame({
        "Aspect": ["Prediction Time", "Accuracy", "Clinical Viability", "Computational Cost", "Scalability"],
        "Deep Learning": ["< 1 second", "MAE < 0.015", "Yes", "Low", "High"],
        "Traditional FE": ["12-24 hours", "Reference", "Limited", "High", "Low"],
    })
    st.table(comparison_table.set_index("Aspect"))

    st.markdown('<p class="mp-section-title">Feature Attribution (Grad-CAM)</p>', unsafe_allow_html=True)
    if ed_final is None or es_final is None:
        st.info("Upload an ED/ES pair in the Interactive Demo tab to view its Grad-CAM attention map.")
    else:
        target = st.radio("Attribution target", ["Ca", "Cb"], horizontal=True, key="gradcam_target")
        heat = compute_real_gradcam(MODEL, ed_final, es_final, output_index=0 if target == "Ca" else 1)
        if heat is None:
            st.warning("Grad-CAM could not be computed — no convolutional layer was found in this model.")
        else:
            from skimage.transform import resize
            display_img = resize(ed_final, (128, 128))
            st.plotly_chart(plot_gradcam_overlay(display_img, heat, "Uploaded Case"), use_container_width=True)
            st.caption(
                "Gradient-based Grad-CAM computed via backpropagation through the "
                "model's last convolutional layer for the selected output."
            )

# =========================================================== TAB 5 =========
with tab5:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="mp-section-title">Clinical Applications</p>', unsafe_allow_html=True)
        st.markdown(
            """<div class="mp-card" style="min-height:200px;">
            <ul>
            <li>Early diagnosis of structural heart disease</li>
            <li>Individualized treatment planning</li>
            <li>Patient risk stratification</li>
            <li>Longitudinal disease progression monitoring</li>
            <li>Personalized surgical planning</li>
            </ul>
            </div>""",
            unsafe_allow_html=True,
        )
        st.markdown('<p class="mp-section-title">Benefits</p>', unsafe_allow_html=True)
        st.markdown(
            """<div class="mp-card" style="min-height:200px;">
            <ul>
            <li>Non-invasive parameter estimation from routine cine MRI</li>
            <li>Near real-time results</li>
            <li>Enables population-scale studies</li>
            <li>Reduces computational burden relative to inverse FE</li>
            <li>Accelerates clinical translation of biomechanical models</li>
            </ul>
            </div>""",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown('<p class="mp-section-title">Limitations</p>', unsafe_allow_html=True)
        st.markdown(
            """<div class="mp-card" style="min-height:200px;">
            <ul>
            <li>Predictions depend on correct ED/ES labeling by the user</li>
            <li>No ground truth exists for uploaded images</li>
            <li>Model training provenance is undocumented</li>
            <li>Validation across scanners and protocols is required</li>
            <li>Limited to single mid-ventricular short-axis slices</li>
            </ul>
            </div>""",
            unsafe_allow_html=True,
        )
        st.markdown('<p class="mp-section-title">Future Directions</p>', unsafe_allow_html=True)
        st.markdown(
            """<div class="mp-card" style="min-height:200px;">
            <ul>
            <li>Validation against expert-labeled clinical data</li>
            <li>Extension to full 3D ventricular geometry</li>
            <li>Incorporation of right-ventricular mechanics</li>
            <li>Physics-informed neural network constraints</li>
            <li>Multi-scanner, multi-site validation</li>
            </ul>
            </div>""",
            unsafe_allow_html=True,
        )
