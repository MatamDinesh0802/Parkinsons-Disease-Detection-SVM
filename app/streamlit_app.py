"""Streamlit demo — Parkinson's Disease Detection from voice features."""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.parkinsons.config import (  # noqa: E402
    CLASS_NAMES,
    FEATURE_COLUMNS,
    METRICS_PATH,
    RAW_DATA,
    REPORTS_DIR,
)
from src.parkinsons.predict import ParkinsonsPredictor  # noqa: E402


# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Parkinson's Detection — Voice Biomarker AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Cached loaders
# ============================================================
def inject_css() -> None:
    css_path = ROOT / "app" / "assets" / "style.css"
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_predictor() -> ParkinsonsPredictor:
    return ParkinsonsPredictor()


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(RAW_DATA)


@st.cache_data(show_spinner=False)
def load_metrics() -> dict | None:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return None


@st.cache_data(show_spinner=False)
def class_means(df: pd.DataFrame) -> tuple[dict, dict]:
    healthy = df[df["status"] == 0][FEATURE_COLUMNS].mean().to_dict()
    parkinsons = df[df["status"] == 1][FEATURE_COLUMNS].mean().to_dict()
    return healthy, parkinsons


@st.cache_data(show_spinner=False)
def top_discriminative_features(df: pd.DataFrame, k: int = 8) -> list[str]:
    h = df[df["status"] == 0][FEATURE_COLUMNS].mean()
    p = df[df["status"] == 1][FEATURE_COLUMNS].mean()
    pooled = df[FEATURE_COLUMNS].std().replace(0, 1)
    score = (h - p).abs() / pooled
    return score.sort_values(ascending=False).head(k).index.tolist()


# ============================================================
# Feature grouping
# ============================================================
FEATURE_GROUPS: dict[str, dict] = {
    "🎙️ Fundamental frequency (Hz)": {
        "features": ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)"],
        "desc": "Average, max, and min vocal fundamental frequency.",
    },
    "📈 Jitter — frequency variation": {
        "features": ["MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP",
                     "MDVP:PPQ", "Jitter:DDP"],
        "desc": "Cycle-to-cycle variation in fundamental frequency.",
    },
    "🎚️ Shimmer — amplitude variation": {
        "features": ["MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3",
                     "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA"],
        "desc": "Cycle-to-cycle variation in signal amplitude.",
    },
    "🔊 Noise / harmonics": {
        "features": ["NHR", "HNR"],
        "desc": "Noise-to-harmonics and harmonics-to-noise ratios.",
    },
    "🌀 Nonlinear dynamics": {
        "features": ["RPDE", "DFA", "D2"],
        "desc": "Signal complexity and self-similarity measures.",
    },
    "📐 Pitch nonlinearity": {
        "features": ["spread1", "spread2", "PPE"],
        "desc": "Fundamental frequency variation measures.",
    },
}

PRESET_DESCRIPTIONS = {
    "Healthy sample": "A real recording from the dataset labelled `status = 0`. Biomarkers fall in typical healthy ranges.",
    "Parkinson's sample": "A real recording from the dataset labelled `status = 1`. Voice shows characteristic Parkinson's signatures.",
    "Dataset mean": "The average of all 195 dataset recordings — a 'centroid' baseline.",
    "Custom": "Enter your own 22 voice biomarkers in the form below.",
}


# ============================================================
# UI components
# ============================================================
def hero(model_name: str | None) -> None:
    badge_model = (model_name or "Not trained").replace("_", " ").title()
    st.markdown(
        f"""
        <div class="hero">
            <p class="hero-eyebrow">Voice-biomarker AI · Portfolio Project</p>
            <div class="hero-title">🧠 Parkinson's Disease Detection</div>
            <p class="hero-subtitle">
                Predict Parkinson's risk from 22 voice biomarkers — fundamental frequency,
                jitter, shimmer, and nonlinear dynamics — using a classical-ML pipeline
                trained on the UCI Parkinson's dataset.
            </p>
            <div class="hero-badges">
                <span class="badge">🎯 Best model: {badge_model}</span>
                <span class="badge">📊 195 voice recordings</span>
                <span class="badge">🧪 5 classifiers benchmarked</span>
                <span class="badge">⚡ &lt;5 ms inference</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_bar(model_name: str, n_features: int, dataset_size: int,
               last_predict_ms: float | None) -> None:
    last = f"{last_predict_ms:.1f} ms" if last_predict_ms is not None else "—"
    st.markdown(
        f"""
        <div class="status-bar">
            <div class="status-item"><span class="status-dot"></span>
                <span class="status-key">Model</span>
                <span class="status-value">{model_name}</span>
            </div>
            <div class="status-item">
                <span class="status-key">Features</span>
                <span class="status-value">{n_features}</span>
            </div>
            <div class="status-item">
                <span class="status-key">Dataset</span>
                <span class="status-value">{dataset_size} recordings</span>
            </div>
            <div class="status-item">
                <span class="status-key">Last inference</span>
                <span class="status-value">{last}</span>
            </div>
            <div class="status-item">
                <span class="status-key">Version</span>
                <span class="status-value">v1.0.0</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def stat_card(col, label: str, value: str, sub: str = "") -> None:
    col.markdown(
        f"""
        <div class="stat-card">
            <p class="stat-label">{label}</p>
            <p class="stat-value">{value}</p>
            <p class="stat-sub">{sub}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def preset_card(preset: str) -> None:
    desc = PRESET_DESCRIPTIONS.get(preset, "")
    st.markdown(
        f"""
        <div class="preset-card">
            <p class="preset-eyebrow">Selected preset</p>
            <p class="preset-name">{preset}</p>
            <p class="preset-desc">{desc}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_card(label: str, prob_pd: float) -> None:
    is_pd = label != "Healthy"
    cls = "parkinsons" if is_pd else "healthy"
    icon = "⚠️" if is_pd else "✅"
    headline = "Parkinson's indicators detected" if is_pd else "No Parkinson's indicators"
    sub = (
        "The voice biomarkers match patterns associated with Parkinson's disease. "
        "This is a screening signal only — not a diagnosis."
        if is_pd
        else "The voice biomarkers fall within healthy ranges for this model."
    )
    st.markdown(
        f"""
        <div class="result-card {cls}">
            <p class="result-label">{icon} Prediction</p>
            <p class="result-headline">{headline}</p>
            <p class="result-prob">Parkinson's probability: <b>{prob_pd*100:.1f}%</b> &nbsp;·&nbsp; {sub}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def empty_state(preset: str) -> None:
    msg = (
        "Adjust the biomarkers in the form below, then run the prediction."
        if preset == "Custom"
        else "A real-world sample is loaded. Hit **Run prediction** to score it."
    )
    st.markdown(
        f"""
        <div class="empty-state">
            <div class="empty-state-icon">🩺</div>
            <p class="empty-state-title">Ready when you are</p>
            <p class="empty-state-text">{msg}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Charts
# ============================================================
def probability_gauge(prob_pd: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_pd * 100,
        number={"suffix": "%", "font": {"size": 38, "family": "Plus Jakarta Sans"},
                "valueformat": ".1f"},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94A3B8",
                     "tickfont": {"size": 10, "color": "#64748B"}},
            "bar": {"color": "#0EA5E9", "thickness": 0.28},
            "bgcolor": "white",
            "borderwidth": 1,
            "bordercolor": "#E2E8F0",
            "steps": [
                {"range": [0, 33], "color": "#DCFCE7"},
                {"range": [33, 66], "color": "#FEF3C7"},
                {"range": [66, 100], "color": "#FECACA"},
            ],
            "threshold": {
                "line": {"color": "#0F172A", "width": 3},
                "thickness": 0.78,
                "value": prob_pd * 100,
            },
        },
        title={"text": "Parkinson's probability",
               "font": {"size": 13, "color": "#475569"}},
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=46, b=10),
                      paper_bgcolor="white")
    return fig


def class_probability_bars(proba: np.ndarray) -> go.Figure:
    labels = [CLASS_NAMES[i] for i in range(len(proba))]
    colors = ["#10B981" if l == "Healthy" else "#F43F5E" for l in labels]
    fig = go.Figure(go.Bar(
        x=[p * 100 for p in proba], y=labels, orientation="h",
        marker=dict(color=colors, line=dict(color="white", width=2)),
        text=[f"{p*100:.1f}%" for p in proba],
        textposition="outside",
        textfont=dict(size=14, family="Plus Jakarta Sans"),
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=20, r=40, t=46, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(range=[0, 110], showgrid=True, gridcolor="#F1F5F9",
                   ticksuffix="%", tickfont=dict(color="#64748B")),
        yaxis=dict(showgrid=False, tickfont=dict(size=13)),
        title=dict(text="Class probabilities",
                   font=dict(size=13, color="#475569")),
    )
    return fig


def comparison_chart(values: dict, healthy: dict, parkinsons: dict,
                     features: list[str]) -> go.Figure:
    user_vals = [values[f] for f in features]
    h_vals = [healthy[f] for f in features]
    p_vals = [parkinsons[f] for f in features]

    # Normalise each feature to [0,1] across the three series so they're visually comparable
    norm = []
    for u, h, p in zip(user_vals, h_vals, p_vals):
        lo, hi = min(u, h, p), max(u, h, p)
        span = hi - lo if hi != lo else 1.0
        norm.append(((u - lo) / span, (h - lo) / span, (p - lo) / span))
    u_n = [n[0] for n in norm]
    h_n = [n[1] for n in norm]
    p_n = [n[2] for n in norm]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Your input", x=features, y=u_n,
                         marker_color="#0EA5E9",
                         hovertemplate="%{x}<br>Your: %{customdata:.4f}<extra></extra>",
                         customdata=user_vals))
    fig.add_trace(go.Bar(name="Healthy avg", x=features, y=h_n,
                         marker_color="#10B981",
                         hovertemplate="%{x}<br>Healthy avg: %{customdata:.4f}<extra></extra>",
                         customdata=h_vals))
    fig.add_trace(go.Bar(name="Parkinson's avg", x=features, y=p_n,
                         marker_color="#F43F5E",
                         hovertemplate="%{x}<br>PD avg: %{customdata:.4f}<extra></extra>",
                         customdata=p_vals))
    fig.update_layout(
        barmode="group",
        height=380,
        margin=dict(l=20, r=20, t=40, b=80),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        title=dict(text="Your input vs class averages (top discriminative features, normalised)",
                   font=dict(size=13, color="#475569")),
        xaxis=dict(tickangle=-30, tickfont=dict(size=10)),
        yaxis=dict(showticklabels=False, showgrid=False),
    )
    return fig


def feature_deviation_chart(user_vec: np.ndarray, df: pd.DataFrame) -> go.Figure:
    means = df[FEATURE_COLUMNS].mean().values
    stds = df[FEATURE_COLUMNS].std().values
    z = (user_vec - means) / np.where(stds == 0, 1, stds)
    order = np.argsort(np.abs(z))[::-1][:10]
    feats = [FEATURE_COLUMNS[i] for i in order]
    zvals = [z[i] for i in order]
    colors = ["#F43F5E" if v > 0 else "#0EA5E9" for v in zvals]

    fig = go.Figure(go.Bar(
        x=zvals, y=feats, orientation="h",
        marker=dict(color=colors),
        hovertemplate="%{y}<br>z-score: %{x:.2f}<extra></extra>",
    ))
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        title=dict(text="Top features deviating from dataset mean (z-score)",
                   font=dict(size=13, color="#475569")),
        xaxis=dict(zeroline=True, zerolinecolor="#94A3B8",
                   showgrid=True, gridcolor="#F1F5F9"),
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="white", plot_bgcolor="white",
    )
    return fig


# ============================================================
# Input rendering
# ============================================================
def render_custom_inputs(df: pd.DataFrame, defaults: dict) -> dict:
    """Render the biomarker form across the main screen in group cards."""
    st.markdown(
        '<div class="section-title">📝 Enter voice biomarkers '
        '<span class="pill">Custom mode</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="section-caption">Inputs are grouped by acoustic family. '
        'Values are pre-filled with dataset means — adjust whatever you have.</p>',
        unsafe_allow_html=True,
    )

    values: dict[str, float] = {}
    group_items = list(FEATURE_GROUPS.items())

    # Render groups in a 2-column outer grid so the page uses space well
    for i in range(0, len(group_items), 2):
        cols = st.columns(2, gap="medium")
        for j, col in enumerate(cols):
            if i + j >= len(group_items):
                continue
            group_name, group_meta = group_items[i + j]
            with col:
                st.markdown(
                    f'<div class="group-card">'
                    f'<p class="group-title">{group_name}</p>',
                    unsafe_allow_html=True,
                )
                st.caption(group_meta["desc"])
                feats = group_meta["features"]
                # 2-column input grid inside each group card
                for k in range(0, len(feats), 2):
                    inner_cols = st.columns(2, gap="small")
                    for m, ic in enumerate(inner_cols):
                        if k + m >= len(feats):
                            continue
                        c = feats[k + m]
                        col_min = float(df[c].min())
                        col_max = float(df[c].max())
                        step = (col_max - col_min) / 200 if col_max > col_min else 0.001
                        values[c] = ic.number_input(
                            c,
                            min_value=col_min - 0.5 * abs(col_min) - 0.001,
                            max_value=col_max + 0.5 * abs(col_max) + 0.001,
                            value=float(defaults[c]),
                            step=step,
                            format="%.5f",
                            key=f"custom__{c}",
                        )
                st.markdown('</div>', unsafe_allow_html=True)
    return values


# ============================================================
# Sidebar
# ============================================================
def sidebar(df: pd.DataFrame) -> tuple[str, dict]:
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-icon">🧠</div>
            <div>
                <div class="sidebar-brand-text">Parkinson's Detection</div>
                <div class="sidebar-brand-sub">Voice biomarker AI · v1.0</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("### 🎛️ Input source")
    preset = st.sidebar.selectbox(
        "Choose a preset or enter your own",
        list(PRESET_DESCRIPTIONS.keys()),
        index=0,
        help="Real samples are pulled from the UCI dataset. 'Custom' opens an editable form on the main screen.",
    )

    if preset == "Healthy sample":
        defaults = {c: float(df[df["status"] == 0].iloc[0][c]) for c in FEATURE_COLUMNS}
    elif preset == "Parkinson's sample":
        defaults = {c: float(df[df["status"] == 1].iloc[0][c]) for c in FEATURE_COLUMNS}
    elif preset == "Dataset mean":
        defaults = {c: float(df[c].mean()) for c in FEATURE_COLUMNS}
    else:
        defaults = {c: float(df[c].mean()) for c in FEATURE_COLUMNS}

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 📚 Links")
    st.sidebar.markdown(
        "- [GitHub repo](https://github.com/MatamDinesh0802/Parkinsons-Disease-Detection-SVM)  \n"
        "- [UCI dataset](https://archive.ics.uci.edu/dataset/174/parkinsons)  \n"
        "- [Original paper](https://doi.org/10.1186/1475-925X-6-23)"
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "**⚠️ Not medical advice.** Research/portfolio demo only. "
        "Always consult a qualified clinician."
    )
    return preset, defaults


# ============================================================
# Main app
# ============================================================
def main() -> None:
    inject_css()

    # Session state defaults
    if "last_prediction" not in st.session_state:
        st.session_state.last_prediction = None
    if "last_predict_ms" not in st.session_state:
        st.session_state.last_predict_ms = None
    if "history" not in st.session_state:
        st.session_state.history = []

    # Predictor
    predictor = None
    model_name = "—"
    try:
        predictor = get_predictor()
        model_name = predictor.model_name
    except FileNotFoundError:
        pass

    hero(model_name if predictor else None)

    if predictor is None:
        st.warning(
            "Model artifacts not found. Run `python -m src.parkinsons.train` first, "
            "then refresh this page."
        )
        st.stop()

    df = load_dataset()
    metrics = load_metrics()
    healthy_mean, pd_mean = class_means(df)
    top_feats = top_discriminative_features(df, k=8)

    status_bar(model_name, len(FEATURE_COLUMNS), len(df), st.session_state.last_predict_ms)

    # ---- Stat cards ----
    c1, c2, c3, c4 = st.columns(4, gap="small")
    stat_card(c1, "Dataset", f"{len(df)}", "voice recordings")
    stat_card(c2, "Features", f"{len(FEATURE_COLUMNS)}", "acoustic biomarkers")
    stat_card(
        c3, "Best model",
        (metrics["best_model"] if metrics else model_name).replace("_", " ").title(),
        "by ROC-AUC",
    )
    if metrics:
        best = metrics["models"][metrics["best_model"]]
        stat_card(c4, "Test ROC-AUC", f"{best['roc_auc']:.3f}",
                  f"Accuracy {best['accuracy']*100:.1f}%")
    else:
        stat_card(c4, "Test ROC-AUC", "—", "Train to populate")

    preset, defaults = sidebar(df)

    tab_pred, tab_expl, tab_perf, tab_about = st.tabs(
        ["🔬 Prediction", "🧠 Explainability", "📊 Model performance", "📖 About"]
    )

    # ============================================================
    # Prediction tab
    # ============================================================
    with tab_pred:
        if preset == "Custom":
            values = render_custom_inputs(df, defaults)
        else:
            preset_card(preset)
            values = defaults

        # Run prediction button — full-width primary
        st.markdown('<div style="height: 0.5rem;"></div>', unsafe_allow_html=True)
        run_col1, run_col2, run_col3 = st.columns([1, 2, 1])
        with run_col2:
            do_predict = st.button(
                "🔮 Run prediction",
                type="primary",
                use_container_width=True,
                key="run_predict",
            )

        if do_predict:
            with st.status("Scoring voice biomarkers…", expanded=False) as status:
                t0 = time.perf_counter()
                status.update(label="Standardising 22 features…")
                pred = predictor.predict(values)
                ms = (time.perf_counter() - t0) * 1000
                status.update(label=f"Done in {ms:.1f} ms", state="complete")

            st.session_state.last_prediction = pred
            st.session_state.last_predict_ms = ms
            st.session_state.history.insert(0, {
                "ts": datetime.now().strftime("%H:%M:%S"),
                "preset": preset,
                "label": pred.label,
                "prob_pd": float(pred.proba_vector[1]),
            })
            st.session_state.history = st.session_state.history[:8]

        # Show result if we have one
        pred = st.session_state.last_prediction
        if pred is not None:
            render_result_card(pred.label, float(pred.proba_vector[1]))

            row1c1, row1c2 = st.columns(2, gap="medium")
            with row1c1:
                st.plotly_chart(probability_gauge(float(pred.proba_vector[1])),
                                use_container_width=True)
            with row1c2:
                st.plotly_chart(class_probability_bars(pred.proba_vector),
                                use_container_width=True)

            st.plotly_chart(
                comparison_chart(values, healthy_mean, pd_mean, top_feats),
                use_container_width=True,
            )

            # Export + history strip
            export_payload = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "preset": preset,
                "model": model_name,
                "prediction": {
                    "label": pred.label,
                    "prob_healthy": float(pred.proba_vector[0]),
                    "prob_parkinsons": float(pred.proba_vector[1]),
                },
                "inputs": values,
            }
            exp_col1, exp_col2 = st.columns([1, 3])
            with exp_col1:
                st.download_button(
                    "⬇️ Download report (JSON)",
                    data=json.dumps(export_payload, indent=2),
                    file_name=f"parkinsons_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True,
                )

            if len(st.session_state.history) > 1:
                st.markdown(
                    '<div class="section-title">🕘 Session history</div>',
                    unsafe_allow_html=True,
                )
                for h in st.session_state.history[1:]:
                    pill_cls = "parkinsons" if h["label"] != "Healthy" else "healthy"
                    st.markdown(
                        f"""<div class="history-card">
                            <span style="color:#94A3B8;font-family:'JetBrains Mono',monospace;">{h['ts']}</span>
                            <span style="color:#475569;">{h['preset']}</span>
                            <span style="margin-left:auto;color:#64748B;">P(PD) {h['prob_pd']*100:.1f}%</span>
                            <span class="history-pill {pill_cls}">{h['label']}</span>
                        </div>""",
                        unsafe_allow_html=True,
                    )
        else:
            empty_state(preset)

    # ============================================================
    # Explainability tab
    # ============================================================
    with tab_expl:
        st.markdown(
            '<div class="section-title">🧠 How does this input compare to the dataset?</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="section-caption">Bars show how far each feature deviates from '
            'the training-set mean, in standard deviations. Large positive (red) or '
            'negative (blue) values are what the model "notices" most.</p>',
            unsafe_allow_html=True,
        )

        user_vec = np.array([values[c] for c in FEATURE_COLUMNS], dtype=float)
        st.plotly_chart(feature_deviation_chart(user_vec, df), use_container_width=True)

        st.markdown(
            '<div class="section-title">📋 Feature value snapshot</div>',
            unsafe_allow_html=True,
        )
        snap = pd.DataFrame({
            "feature": FEATURE_COLUMNS,
            "your value": user_vec,
            "healthy mean": [healthy_mean[c] for c in FEATURE_COLUMNS],
            "parkinson's mean": [pd_mean[c] for c in FEATURE_COLUMNS],
            "dataset std": df[FEATURE_COLUMNS].std().values,
        })
        st.dataframe(snap, hide_index=True, use_container_width=True)

    # ============================================================
    # Performance tab
    # ============================================================
    with tab_perf:
        if metrics is None:
            st.warning("Run training first to populate metrics.")
        else:
            rows = [{"model": n, **m} for n, m in metrics["models"].items()]
            mdf = pd.DataFrame(rows).set_index("model")
            st.markdown(
                '<div class="section-title">📊 Held-out test metrics</div>',
                unsafe_allow_html=True,
            )
            st.dataframe(
                mdf.style.format("{:.4f}").background_gradient(cmap="Blues", axis=0),
                use_container_width=True,
            )

            col1, col2 = st.columns(2, gap="medium")
            fig_roc = REPORTS_DIR / "figures" / "roc_curves.png"
            fig_cm = REPORTS_DIR / "figures" / f"confusion_matrix_{metrics['best_model']}.png"
            with col1:
                if fig_roc.exists():
                    st.image(str(fig_roc), caption="ROC curves — all models")
            with col2:
                if fig_cm.exists():
                    st.image(str(fig_cm),
                             caption=f"Confusion matrix — {metrics['best_model']}")

            fig_bar = REPORTS_DIR / "figures" / "model_comparison.png"
            if fig_bar.exists():
                st.image(str(fig_bar), caption="Model comparison across metrics")

    # ============================================================
    # About tab
    # ============================================================
    with tab_about:
        col_a, col_b = st.columns([2, 1], gap="large")
        with col_a:
            st.markdown("""
### Problem
Parkinson's disease causes characteristic disturbances in voice — reduced loudness,
breathiness, monotone pitch, and tremor. Vocal biomarkers can act as a low-cost,
non-invasive **screening signal** before clinical assessment.

### Approach
1. **Data**: UCI Parkinson's dataset — 195 voice recordings from 31 subjects, 22 acoustic features.
2. **Pipeline**: stratified train/test split → `StandardScaler` → classifier.
3. **Models**: SVM (linear, RBF) headline, with Logistic Regression, Random Forest, and Gradient Boosting baselines.
4. **Selection**: best model is chosen by held-out ROC-AUC.

### Limitations
- The dataset is small (195 rows) and from a single recording protocol.
- This is not a medical device — only a research/portfolio demonstration.
- Predictions on out-of-distribution voice data (different mic, language, accent) may be unreliable.
""")
        with col_b:
            st.markdown("""
### Tech stack
- **scikit-learn** — classical ML
- **Plotly** — interactive viz
- **Streamlit** — this UI

### Links
- [UCI Parkinson dataset](https://archive.ics.uci.edu/dataset/174/parkinsons)
- [GitHub repo](https://github.com/MatamDinesh0802/Parkinsons-Disease-Detection-SVM)

### Author
**Matam Dinesh**
[matamdinesh0802@gmail.com](mailto:matamdinesh0802@gmail.com)
""")

    # ============================================================
    # Footer
    # ============================================================
    st.markdown(
        f"""
        <div class="footer">
            <div>© 2026 Matam Dinesh · MIT License · Built with Streamlit & scikit-learn</div>
            <div>
                <a href="https://github.com/MatamDinesh0802/Parkinsons-Disease-Detection-SVM">GitHub</a>
                · <a href="mailto:matamdinesh0802@gmail.com">Contact</a>
                · <a href="https://archive.ics.uci.edu/dataset/174/parkinsons">Dataset</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
