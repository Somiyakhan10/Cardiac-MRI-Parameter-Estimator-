import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

from utils.data_loader import PATHOLOGY_COLORS, PATHOLOGY_ORDER

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

SERIES_PRED = "#2a78d6"
SERIES_GT = "#008300"

FONT = dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", color=INK_PRIMARY)

def _base_layout(title, height=380, **kwargs):
    layout = dict(
        title=dict(text=title, font=dict(size=16, color=INK_PRIMARY)),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=FONT,
        height=height,
        margin=dict(l=50, r=30, t=60, b=50),
        hoverlabel=dict(bgcolor="white", font_size=13, font_family=FONT["family"]),
    )
    layout.update(kwargs)
    return layout

def _axis(title=""):
    return dict(
        title=dict(text=title, font=dict(size=12, color=INK_SECONDARY)),
        gridcolor=GRIDLINE,
        zerolinecolor=BASELINE,
        linecolor=BASELINE,
        tickfont=dict(color=INK_MUTED, size=11),
    )

def plot_mri_pair(ed_img: np.ndarray, es_img: np.ndarray, patient_id: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=np.flipud(ed_img), colorscale="gray", showscale=False,
                              x0=0, dx=1, xaxis="x", yaxis="y",
                              hovertemplate="ED intensity: %{z:.2f}<extra></extra>"))
    fig.add_trace(go.Heatmap(z=np.flipud(es_img), colorscale="gray", showscale=False,
                              xaxis="x2", yaxis="y2",
                              hovertemplate="ES intensity: %{z:.2f}<extra></extra>"))
    fig.update_layout(
        **_base_layout(f"{patient_id} — Mid-Ventricular Short-Axis Slice", height=340),
        xaxis=dict(domain=[0, 0.47], visible=False, scaleanchor="y"),
        yaxis=dict(domain=[0, 1], visible=False),
        xaxis2=dict(domain=[0.53, 1], visible=False, scaleanchor="y2"),
        yaxis2=dict(domain=[0, 1], visible=False),
        annotations=[
            dict(text="End-Diastole (ED)", x=0.235, y=1.06, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=13, color=INK_SECONDARY)),
            dict(text="End-Systole (ES)", x=0.765, y=1.06, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=13, color=INK_SECONDARY)),
        ],
        showlegend=False,
    )
    return fig

def plot_param_comparison_bar(record: dict) -> go.Figure:
    params = ["Ca", "Cb"]
    pred = [record["ca_pred"], record["cb_pred"]]
    gt = [record["ca_gt"], record["cb_gt"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Predicted", x=params, y=pred, marker_color=SERIES_PRED,
                          text=[f"{v:.3f}" for v in pred], textposition="outside",
                          hovertemplate="Predicted %{x}: %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Bar(name="Ground Truth", x=params, y=gt, marker_color=SERIES_GT,
                          text=[f"{v:.3f}" for v in gt], textposition="outside",
                          hovertemplate="Ground Truth %{x}: %{y:.3f}<extra></extra>"))
    fig.update_layout(
        **_base_layout("Predicted vs. Ground Truth Parameters", height=380),
        barmode="group",
        bargap=0.35,
        yaxis=_axis("Scaling factor value"),
        xaxis=_axis(""),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig

def plot_prediction_bar(ca_pred: float, cb_pred: float, ca_std: float = None, cb_std: float = None) -> go.Figure:
    params = ["Ca", "Cb"]
    values = [ca_pred, cb_pred]

    bar_kwargs = dict(
        x=params, y=values, marker_color=SERIES_PRED,
        text=[f"{v:.3f}" for v in values], textposition="outside",
        hovertemplate="Predicted %{x}: %{y:.3f}<extra></extra>",
    )
    if ca_std is not None and cb_std is not None:
        bar_kwargs["error_y"] = dict(type="data", array=[ca_std, cb_std], visible=True, color=INK_SECONDARY)

    fig = go.Figure(go.Bar(**bar_kwargs))
    fig.update_layout(
        **_base_layout("Predicted Parameters (mean ± SD, Monte Carlo Dropout)", height=380),
        bargap=0.5,
        yaxis=_axis("Scaling factor value"),
        xaxis=_axis(""),
        showlegend=False,
    )
    return fig


def plot_speed_comparison() -> go.Figure:
    labels = ["ML Prediction", "Traditional FE Optimization"]
    seconds = [0.08, 14.5 * 3600]
    colors = [SERIES_PRED, "#eb6834"]

    fig = go.Figure(go.Bar(
        x=labels, y=seconds, marker_color=colors,
        text=["0.08 sec", "14.5 hours"], textposition="outside",
        hovertemplate="%{x}: %{y:.0f} sec<extra></extra>",
    ))
    fig.update_layout(
        **_base_layout("Inference Time: ML vs. Traditional Inverse FE (log scale)", height=380),
        yaxis=dict(**_axis("Time (seconds, log scale)"), type="log"),
        xaxis=_axis(""),
        showlegend=False,
    )
    return fig

def plot_scatter_pred_vs_gt(df: pd.DataFrame, param: str = "ca") -> go.Figure:
    gt_col, pred_col = f"{param}_gt", f"{param}_pred"
    fig = go.Figure()

    for group in PATHOLOGY_ORDER:
        sub = df[df["pathology"] == group]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub[gt_col], y=sub[pred_col], mode="markers", name=group,
            marker=dict(color=PATHOLOGY_COLORS[group], size=11, line=dict(width=1, color="white")),
            customdata=sub["patient_id"],
            hovertemplate=f"%{{customdata}}<br>GT {param.upper()}: %{{x:.3f}}<br>Pred {param.upper()}: %{{y:.3f}}<extra></extra>",
        ))

    x_all = df[gt_col].values.reshape(-1, 1)
    y_all = df[pred_col].values
    reg = LinearRegression().fit(x_all, y_all)
    r2 = reg.score(x_all, y_all)
    x_line = np.linspace(x_all.min(), x_all.max(), 50)
    y_line = reg.predict(x_line.reshape(-1, 1))
    fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", name="Regression fit",
                              line=dict(color=INK_SECONDARY, dash="dash", width=2),
                              hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x_line, y=x_line, mode="lines", name="Identity (y=x)",
                              line=dict(color=BASELINE, dash="dot", width=1.5),
                              hoverinfo="skip"))

    fig.update_layout(
        **_base_layout(f"Predicted vs. Ground Truth — {param.upper()}", height=440),
        xaxis=_axis(f"Ground Truth {param.upper()}"),
        yaxis=_axis(f"Predicted {param.upper()}"),
        legend=dict(title="Pathology"),
        annotations=[dict(
            text=f"R² = {r2:.2f}", x=0.04, y=0.96, xref="paper", yref="paper",
            showarrow=False, font=dict(size=14, color=INK_PRIMARY),
            bgcolor="rgba(255,255,255,0.7)",
        )],
    )
    return fig

def plot_bland_altman(df: pd.DataFrame, param: str = "ca") -> go.Figure:
    gt_col, pred_col = f"{param}_gt", f"{param}_pred"
    mean_vals = (df[gt_col] + df[pred_col]) / 2
    diff_vals = df[pred_col] - df[gt_col]
    bias = diff_vals.mean()
    sd = diff_vals.std()
    upper_loa = bias + 1.96 * sd
    lower_loa = bias - 1.96 * sd

    fig = go.Figure()
    for group in PATHOLOGY_ORDER:
        mask = df["pathology"] == group
        if not mask.any():
            continue
        fig.add_trace(go.Scatter(
            x=mean_vals[mask], y=diff_vals[mask], mode="markers", name=group,
            marker=dict(color=PATHOLOGY_COLORS[group], size=11, line=dict(width=1, color="white")),
            customdata=df.loc[mask, "patient_id"],
            hovertemplate=f"%{{customdata}}<br>Mean: %{{x:.3f}}<br>Diff: %{{y:.3f}}<extra></extra>",
        ))

    x_range = [mean_vals.min() - 0.02, mean_vals.max() + 0.02]
    for y_val, label, dash in [(bias, f"Bias {bias:.3f}", "solid"),
                               (upper_loa, f"+1.96 SD ({upper_loa:.3f})", "dash"),
                               (lower_loa, f"-1.96 SD ({lower_loa:.3f})", "dash")]:
        fig.add_trace(go.Scatter(x=x_range, y=[y_val, y_val], mode="lines", name=label,
                                  line=dict(color=INK_SECONDARY, dash=dash, width=1.5),
                                  hoverinfo="skip"))

    fig.update_layout(
        **_base_layout(f"Bland-Altman Agreement — {param.upper()}", height=440),
        xaxis=_axis(f"Mean of Predicted & Ground Truth {param.upper()}"),
        yaxis=_axis("Difference (Predicted − Ground Truth)"),
        legend=dict(title="Pathology / Reference"),
    )
    return fig

def plot_mae_by_pathology(mae_dict: dict, param_label: str = "Ca") -> go.Figure:
    groups = list(mae_dict.keys())
    values = list(mae_dict.values())
    colors = [PATHOLOGY_COLORS[g] for g in groups]

    fig = go.Figure(go.Bar(
        x=groups, y=values, marker_color=colors,
        text=[f"{v:.3f}" for v in values], textposition="outside",
        hovertemplate="%{x}: MAE %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        **_base_layout(f"Mean Absolute Error by Pathology Group ({param_label})", height=380),
        yaxis=_axis("MAE"),
        xaxis=_axis("Pathology group"),
        showlegend=False,
    )
    return fig

SEG_STRUCT_COLORS = {
    1: (74, 58, 167, 150),
    2: (235, 104, 52, 150),
    3: (227, 73, 72, 160),
}
SEG_STRUCT_NAMES = {1: "RV cavity", 2: "Myocardium", 3: "LV cavity"}

def plot_segmentation_overlay(image: np.ndarray, seg: np.ndarray, patient_id: str, phase_label: str) -> go.Figure:
    overlay = np.zeros((*seg.shape, 4), dtype=np.uint8)
    for label, color in SEG_STRUCT_COLORS.items():
        overlay[seg == label] = color

    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=np.flipud(image), colorscale="gray", showscale=False, hoverinfo="skip"))
    fig.add_trace(go.Image(z=np.flipud(overlay), hoverinfo="skip"))
    fig.update_layout(
        **_base_layout(f"{patient_id} — {phase_label} Segmentation Overlay", height=420),
        xaxis=dict(visible=False, scaleanchor="y"),
        yaxis=dict(visible=False),
        showlegend=False,
        annotations=[
            dict(text="RV cavity (violet) · Myocardium (orange) · LV cavity (red)", x=0.5, y=-0.06,
                 xref="paper", yref="paper", showarrow=False,
                 font=dict(size=12, color=INK_SECONDARY)),
        ],
    )
    return fig

def plot_gradcam_overlay(image: np.ndarray, heatmap: np.ndarray, patient_id: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=np.flipud(image), colorscale="gray", showscale=False,
                              hoverinfo="skip"))
    fig.add_trace(go.Heatmap(z=np.flipud(heatmap), colorscale="Turbo", opacity=0.55,
                              zmin=0, zmax=1,
                              colorbar=dict(title="Attention", tickfont=dict(color=INK_MUTED)),
                              hovertemplate="Attention: %{z:.2f}<extra></extra>"))
    fig.update_layout(
        **_base_layout(f"Grad-CAM Attention Map — {patient_id} (ED)", height=420),
        xaxis=dict(visible=False, scaleanchor="y"),
        yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig
