from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from config import SYSTEM_NAME

ROOT = Path(__file__).resolve().parent
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Source Sans 3, Segoe UI, sans-serif", color="#0F172A", size=13),
    margin=dict(l=16, r=16, t=48, b=16),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    colorway=["#0F766E", "#1D4ED8", "#C2410C", "#7C3AED", "#0EA5E9", "#B91C1C"],
)


def inject_css():
    css = (ROOT / "assets" / "style.css").read_text()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def page_header(kicker: str, title: str, lede: str = ""):
    extra = f'<div class="page-lede">{lede}</div>' if lede else ""
    st.markdown(
        f'<div class="page-kicker">{kicker}</div><h1 class="page-title">{title}</h1>{extra}',
        unsafe_allow_html=True,
    )


def kpi_cards(items: list[dict]):
    cards = []
    for item in items:
        tone = item.get("tone", "plain")
        cards.append(
            f'<div class="kpi {tone}"><div class="label">{item["label"]}</div>'
            f'<div class="value">{item["value"]}</div></div>'
        )
    st.markdown(f'<div class="kpi-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def risk_badge(band: str) -> str:
    cls = {"High": "badge-high", "Moderate": "badge-moderate", "Low": "badge-low"}.get(band, "badge-low")
    return f'<span class="badge {cls}">{band} risk</span>'


def style_fig(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(**PLOTLY_LAYOUT, height=height)
    fig.update_xaxes(gridcolor="#E2E8F0", zerolinecolor="#CBD5E1")
    fig.update_yaxes(gridcolor="#E2E8F0", zerolinecolor="#CBD5E1")
    return fig


def chart(fig: go.Figure, height: int = 380):
    st.plotly_chart(style_fig(fig, height), use_container_width=True)


def dashboard_summary(
    patients: int,
    predictions: int,
    high: int,
    moderate: int,
    low: int,
    best_model: str,
    roc_auc: float,
    accuracy: float,
    top_factor: str,
):
    st.markdown(
        f"""
        <div class="dash-panel">
          <div class="dash-title">{SYSTEM_NAME}</div>
          <div class="dash-sub">Clinical dashboard — live counts from this workspace</div>
          <div class="dash-grid">
            <div class="dash-stat"><span>Patients assessed</span><strong>{patients:,}</strong></div>
            <div class="dash-stat"><span>OA predictions</span><strong>{predictions:,}</strong></div>
            <div class="dash-stat"><span>High risk</span><strong>{high:,}</strong></div>
            <div class="dash-stat"><span>Moderate risk</span><strong>{moderate:,}</strong></div>
            <div class="dash-stat"><span>Low risk</span><strong>{low:,}</strong></div>
            <div class="dash-stat"><span>Best model</span><strong>{best_model}</strong></div>
            <div class="dash-stat"><span>Test ROC-AUC</span><strong>{roc_auc:.3f}</strong></div>
            <div class="dash-stat"><span>Test accuracy</span><strong>{accuracy:.3f}</strong></div>
            <div class="dash-stat wide"><span>Most important OA risk factor (mean |SHAP|)</span><strong>{top_factor}</strong></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def prediction_outcome(band: str, probability: float, model_name: str):
    st.markdown(
        f"""
        <div class="pred-panel">
          <div class="pred-label">OA risk prediction</div>
          <div class="pred-band">{band.upper()} OA RISK</div>
          <div class="pred-prob">Probability: {probability * 100:.1f}%</div>
          <div class="pred-model">Model: {model_name}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_threshold_legend(low_max: float, moderate_max: float, youden: float):
    st.markdown(
        f"""
        <div class="threshold-panel">
          <strong>Validation-derived risk bands</strong> (not arbitrary cut-offs):
          <ul>
            <li><b>Low</b>: ≤ {low_max * 100:.1f}% (highest threshold with sensitivity ≥ 80% on validation)</li>
            <li><b>Moderate</b>: {low_max * 100:.1f}% – {moderate_max * 100:.1f}%</li>
            <li><b>High</b>: ≥ {moderate_max * 100:.1f}% (lowest threshold with specificity ≥ 80% on validation)</li>
          </ul>
          Binary fairness/decision threshold (Youden's J on validation): {youden * 100:.1f}%
        </div>
        """,
        unsafe_allow_html=True,
    )
