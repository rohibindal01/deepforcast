"""
app.py
Streamlit frontend for the Time Series Forecasting system.
Provides: data exploration, model training, forecasting dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import json
import time
from datetime import datetime

# ─── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="DeepForecast · Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');

    :root {
        --bg: #0a0e1a;
        --card: #111827;
        --border: #1f2937;
        --accent: #00d4ff;
        --accent2: #7c3aed;
        --green: #10b981;
        --red: #ef4444;
        --text: #e2e8f0;
        --muted: #6b7280;
    }

    .stApp { background: var(--bg); }

    .main-header {
        font-family: 'Space Mono', monospace;
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00d4ff 0%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-family: 'Inter', sans-serif;
        color: #6b7280;
        font-size: 0.95rem;
        margin-bottom: 2rem;
        letter-spacing: 0.05em;
    }

    .metric-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.4rem;
    }
    .metric-value {
        font-family: 'Space Mono', monospace;
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--accent);
    }
    .metric-value.green { color: var(--green); }
    .metric-value.red { color: var(--red); }

    .forecast-tag {
        display: inline-block;
        background: linear-gradient(135deg, #00d4ff22, #7c3aed22);
        border: 1px solid #00d4ff44;
        border-radius: 6px;
        padding: 0.3rem 0.8rem;
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        color: var(--accent);
        margin-right: 0.5rem;
    }

    .section-title {
        font-family: 'Space Mono', monospace;
        font-size: 1rem;
        color: var(--accent);
        text-transform: uppercase;
        letter-spacing: 0.12em;
        border-left: 3px solid var(--accent);
        padding-left: 0.8rem;
        margin-bottom: 1rem;
    }

    div[data-testid="stSidebar"] {
        background: #0d1220;
        border-right: 1px solid var(--border);
    }

    .stButton > button {
        background: linear-gradient(135deg, #00d4ff, #7c3aed);
        color: white;
        border: none;
        border-radius: 8px;
        font-family: 'Space Mono', monospace;
        font-size: 0.85rem;
        padding: 0.6rem 1.4rem;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    .info-box {
        background: #111827;
        border: 1px solid #1f2937;
        border-left: 4px solid #00d4ff;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        font-family: 'Inter', sans-serif;
        font-size: 0.88rem;
        color: #94a3b8;
    }

    .warning-box {
        background: #1a1500;
        border: 1px solid #2a2000;
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        font-family: 'Inter', sans-serif;
        font-size: 0.88rem;
        color: #fbbf24;
    }
</style>
""", unsafe_allow_html=True)

# ─── Imports (deferred to avoid slow startup) ───────────────
@st.cache_resource
def import_modules():
    from utils.data_loader import fetch_stock_data, prepare_features
    from predict import predict_future, load_training_history
    return fetch_stock_data, prepare_features, predict_future, load_training_history

# ─── Plotly theme ────────────────────────────────────────────
PLOT_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#94a3b8"),
    xaxis=dict(gridcolor="#1f2937", showgrid=True, zeroline=False),
    yaxis=dict(gridcolor="#1f2937", showgrid=True, zeroline=False),
    margin=dict(l=10, r=10, t=40, b=10),
)

# ─── Header ─────────────────────────────────────────────────
st.markdown('<div class="main-header">DEEPFORECAST</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">LSTM · TRANSFORMER · TEMPORAL CNN — Advanced Stock Price Forecasting</div>',
    unsafe_allow_html=True
)

# ─── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    ticker = st.text_input(
        "Stock Ticker", value="AAPL",
        help="Enter a valid Yahoo Finance ticker (e.g., AAPL, GOOGL, TSLA, ^NSEI)"
    ).upper().strip()

    st.markdown("**Training Settings**")
    period = st.selectbox("Historical Period", ["2y", "3y", "5y", "10y"], index=2)
    seq_length = st.slider("Sequence Length (days)", 30, 120, 60, 10,
                           help="Number of past days used for prediction")
    horizon = st.slider("Forecast Horizon (days)", 3, 30, 7,
                        help="Number of future days to predict")

    st.markdown("**Model Hyperparameters**")
    epochs = st.slider("Max Epochs", 20, 200, 100, 10)
    batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
    lr = st.select_slider(
        "Learning Rate",
        options=[1e-4, 5e-4, 1e-3, 3e-3],
        value=1e-3,
        format_func=lambda x: f"{x:.0e}"
    )

    st.markdown("---")
    train_btn = st.button("🚀 Train Model", use_container_width=True)
    predict_btn = st.button("🔮 Run Forecast", use_container_width=True)

    st.markdown("---")
    st.markdown(
        '<div class="info-box">Model saves to <code>models/saved/</code>.<br>'
        'Train once, forecast anytime.</div>',
        unsafe_allow_html=True
    )

    MODEL_DIR = "models/saved"
    model_exists = os.path.exists(os.path.join(MODEL_DIR, "forecaster.keras"))
    if model_exists:
        st.success("✅ Trained model found")
    else:
        st.warning("⚠️ No model yet — train first")

# ─── Tabs ────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Market Data",
    "🧠 Train Model",
    "🔮 Forecast",
    "📈 Model Analysis",
])

fetch_stock_data, prepare_features, predict_future, load_training_history = import_modules()

# ═══════════════════════════════════════════════════════════════
# TAB 1 — Market Data Explorer
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">Market Data Explorer</div>', unsafe_allow_html=True)

    @st.cache_data(ttl=300)
    def load_data(t, p):
        return fetch_stock_data(t, period=p)

    try:
        with st.spinner(f"Fetching {ticker} data..."):
            df = load_data(ticker, period)

        # KPI Row
        last = df["Close"].iloc[-1]
        prev = df["Close"].iloc[-2]
        chg = last - prev
        chg_pct = chg / prev * 100

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Last Close</div>
                <div class="metric-value">${last:.2f}</div></div>""", unsafe_allow_html=True)
        with col2:
            color = "green" if chg >= 0 else "red"
            sign = "+" if chg >= 0 else ""
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Day Change</div>
                <div class="metric-value {color}">{sign}{chg:.2f}</div></div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Change %</div>
                <div class="metric-value {color}">{sign}{chg_pct:.2f}%</div></div>""", unsafe_allow_html=True)
        with col4:
            high_52 = df["High"].tail(252).max()
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">52W High</div>
                <div class="metric-value">${high_52:.2f}</div></div>""", unsafe_allow_html=True)
        with col5:
            low_52 = df["Low"].tail(252).min()
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">52W Low</div>
                <div class="metric-value">${low_52:.2f}</div></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Candlestick + Volume chart
        display_days = st.slider("Display last N days", 60, min(len(df), 1000), 180, 30)
        plot_df = df.tail(display_days)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.72, 0.28], vertical_spacing=0.02)

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=plot_df.index, open=plot_df["Open"], high=plot_df["High"],
            low=plot_df["Low"], close=plot_df["Close"],
            increasing_line_color="#10b981", decreasing_line_color="#ef4444",
            increasing_fillcolor="#10b98133", decreasing_fillcolor="#ef444433",
            name="OHLC"
        ), row=1, col=1)

        # Moving Averages
        for ma, color in [("MA_7", "#00d4ff"), ("MA_21", "#f59e0b"), ("MA_50", "#7c3aed")]:
            if ma in plot_df.columns:
                fig.add_trace(go.Scatter(
                    x=plot_df.index, y=plot_df[ma],
                    line=dict(color=color, width=1.2),
                    name=ma, opacity=0.8
                ), row=1, col=1)

        # Volume
        vol_colors = ["#10b981" if c >= o else "#ef4444"
                      for c, o in zip(plot_df["Close"], plot_df["Open"])]
        fig.add_trace(go.Bar(
            x=plot_df.index, y=plot_df["Volume"],
            marker_color=vol_colors, opacity=0.6, name="Volume"
        ), row=2, col=1)

        fig.update_layout(**PLOT_THEME, height=580, showlegend=True,
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                          title=dict(text=f"{ticker} · Price & Volume", font_size=14))
        fig.update_xaxes(rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # Technical Indicators
        with st.expander("📉 Technical Indicators"):
            fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                 row_heights=[0.5, 0.5], vertical_spacing=0.08,
                                 subplot_titles=["RSI (14)", "MACD"])
            fig2.add_trace(go.Scatter(x=plot_df.index, y=plot_df["RSI"],
                                      line=dict(color="#00d4ff", width=1.5), name="RSI"), row=1, col=1)
            fig2.add_hline(y=70, line_dash="dot", line_color="#ef4444", row=1, col=1)
            fig2.add_hline(y=30, line_dash="dot", line_color="#10b981", row=1, col=1)

            fig2.add_trace(go.Scatter(x=plot_df.index, y=plot_df["MACD"],
                                      line=dict(color="#7c3aed", width=1.5), name="MACD"), row=2, col=1)
            fig2.add_trace(go.Scatter(x=plot_df.index, y=plot_df["MACD_Signal"],
                                      line=dict(color="#f59e0b", width=1.2), name="Signal"), row=2, col=1)
            hist = plot_df["MACD"] - plot_df["MACD_Signal"]
            fig2.add_trace(go.Bar(x=plot_df.index, y=hist,
                                  marker_color=["#10b981" if v >= 0 else "#ef4444" for v in hist],
                                  opacity=0.6, name="MACD Hist"), row=2, col=1)

            fig2.update_layout(**PLOT_THEME, height=380)
            st.plotly_chart(fig2, use_container_width=True)

        with st.expander("📋 Raw Data"):
            st.dataframe(df.tail(50).style.background_gradient(subset=["Close"], cmap="RdYlGn"),
                         use_container_width=True)

    except Exception as e:
        st.error(f"Error fetching data for **{ticker}**: {e}")
        st.info("Try tickers like: AAPL, GOOGL, MSFT, TSLA, AMZN, ^NSEI, BTC-USD")

# ═══════════════════════════════════════════════════════════════
# TAB 2 — Training
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Model Training</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <b>Architecture:</b> TCN Encoder → Dual Transformer Blocks → Bidirectional LSTM → Temporal Attention → Dense Head<br>
    Training uses <b>Huber Loss</b> (robust to price outliers), <b>Adam</b> optimizer with gradient clipping,
    Early Stopping, and ReduceLROnPlateau scheduling.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1])
    with col_l:
        st.markdown("**Training Configuration**")
        st.markdown(f"""
        | Parameter | Value |
        |-----------|-------|
        | Ticker | `{ticker}` |
        | Historical Period | `{period}` |
        | Sequence Length | `{seq_length} days` |
        | Forecast Horizon | `{horizon} days` |
        | Max Epochs | `{epochs}` |
        | Batch Size | `{batch_size}` |
        | Learning Rate | `{lr:.0e}` |
        """)

    with col_r:
        st.markdown("**Model Architecture Summary**")
        st.code("""
Input(seq_len, n_features)
  ↓
TCNBlock(64, dilation=1)   ← causal conv
TCNBlock(64, dilation=2)   ← wider receptive field
TCNBlock(128, dilation=4)  ← global context
  ↓
Dense(128) → LayerNorm     ← projection
  ↓
TransformerBlock × 2       ← multi-head attention
  ↓
BiLSTM(128) → BiLSTM(64)   ← sequential memory
  ↓
Temporal Attention Pool    ← focus mechanism
  ↓
Dense(256) → Dense(128)    ← prediction head
  ↓
Dense(forecast_horizon)    ← output
""", language="text")

    if train_btn:
        st.markdown("---")
        progress_bar = st.progress(0, text="Initializing training pipeline...")
        log_area = st.empty()
        status_text = st.empty()

        try:
            import subprocess
            import sys

            cmd = [
                sys.executable, "train.py",
                "--ticker", ticker,
                "--period", period,
                "--seq_length", str(seq_length),
                "--horizon", str(horizon),
                "--epochs", str(epochs),
                "--batch_size", str(batch_size),
                "--lr", str(lr),
                "--model_dir", MODEL_DIR,
            ]

            progress_bar.progress(10, text="Fetching and preprocessing data...")

            with st.spinner("Training in progress — this may take a few minutes..."):
                result = subprocess.run(
                    cmd, capture_output=True, text=True,
                    cwd=os.path.dirname(os.path.abspath(__file__)) or "."
                )

            progress_bar.progress(100, text="Done!")

            if result.returncode == 0:
                st.success("✅ Training completed successfully!")
                with st.expander("📄 Training Log", expanded=True):
                    st.code(result.stdout, language="text")

                # Load and show training history
                hist_df = load_training_history(MODEL_DIR)
                if hist_df is not None:
                    st.markdown('<div class="section-title" style="margin-top:1rem">Training Curves</div>',
                                unsafe_allow_html=True)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=hist_df["loss"], name="Train Loss",
                                            line=dict(color="#00d4ff", width=2)))
                    if "val_loss" in hist_df.columns:
                        fig.add_trace(go.Scatter(y=hist_df["val_loss"], name="Val Loss",
                                                 line=dict(color="#7c3aed", width=2)))
                    fig.update_layout(**PLOT_THEME, height=300,
                                      title="Training / Validation Loss",
                                      xaxis_title="Epoch", yaxis_title="Huber Loss")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Training failed. See error log below.")
                st.code(result.stderr, language="text")

        except Exception as e:
            st.error(f"Error during training: {e}")
            st.exception(e)

# ═══════════════════════════════════════════════════════════════
# TAB 3 — Forecast Dashboard
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Price Forecast Dashboard</div>', unsafe_allow_html=True)

    if not model_exists:
        st.markdown(
            '<div class="warning-box">⚠️ No trained model found. '
            'Please go to <b>Train Model</b> tab and train first.</div>',
            unsafe_allow_html=True
        )
    else:
        run_forecast = predict_btn or st.button("▶️ Generate Forecast Now")

        if run_forecast:
            try:
                with st.spinner(f"Running inference for {ticker}..."):
                    forecast_df, meta, history_df = predict_future(ticker, MODEL_DIR)

                # ── KPI Cards ──────────────────────────────
                last_price = meta["last_close"]
                next_price = meta["predicted_next"]
                change_pct = meta["predicted_change_pct"]
                direction = "↑" if change_pct >= 0 else "↓"
                color_class = "green" if change_pct >= 0 else "red"

                st.markdown("<br>", unsafe_allow_html=True)
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">Last Close</div>
                        <div class="metric-value">${last_price:.2f}</div></div>""",
                        unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">Next Day Forecast</div>
                        <div class="metric-value {color_class}">${next_price:.2f}</div></div>""",
                        unsafe_allow_html=True)
                with c3:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">Expected Change</div>
                        <div class="metric-value {color_class}">{direction} {abs(change_pct):.2f}%</div></div>""",
                        unsafe_allow_html=True)
                with c4:
                    end_price = float(forecast_df["Predicted_Close"].iloc[-1])
                    total_chg = (end_price - last_price) / last_price * 100
                    sign2 = "+" if total_chg >= 0 else ""
                    cl2 = "green" if total_chg >= 0 else "red"
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">{meta['forecast_horizon']}d Outlook</div>
                        <div class="metric-value {cl2}">{sign2}{total_chg:.2f}%</div></div>""",
                        unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Forecast Chart ─────────────────────────
                recent_hist = history_df.tail(90)
                fig = go.Figure()

                # Historical
                fig.add_trace(go.Scatter(
                    x=recent_hist.index, y=recent_hist["Close"],
                    line=dict(color="#94a3b8", width=1.5),
                    name="Historical Close", mode="lines"
                ))

                # Forecast
                fig.add_trace(go.Scatter(
                    x=forecast_df.index, y=forecast_df["Predicted_Close"],
                    line=dict(color="#00d4ff", width=2.5, dash="dash"),
                    mode="lines+markers",
                    marker=dict(size=7, color="#00d4ff", symbol="circle"),
                    name="Forecast"
                ))

                # Confidence band (±5% heuristic band)
                upper = forecast_df["Predicted_Close"] * 1.05
                lower = forecast_df["Predicted_Close"] * 0.95
                fig.add_trace(go.Scatter(
                    x=list(forecast_df.index) + list(forecast_df.index[::-1]),
                    y=list(upper) + list(lower[::-1]),
                    fill="toself",
                    fillcolor="rgba(0, 212, 255, 0.08)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="±5% Band", showlegend=True
                ))

                # Vertical separator
                last_date = recent_hist.index[-1]
                fig.add_vline(x=str(last_date.date()), line_dash="dot",
                              line_color="#f59e0b", opacity=0.6)
                fig.add_annotation(x=str(last_date.date()), y=last_price,
                                   text=" Today", showarrow=False,
                                   font=dict(color="#f59e0b", size=11),
                                   xanchor="left")

                fig.update_layout(
                    **PLOT_THEME, height=460,
                    title=dict(text=f"{ticker} · {meta['forecast_horizon']}-Day Price Forecast",
                               font_size=14),
                    xaxis_title="Date", yaxis_title="Price (USD)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)

                # ── Forecast Table ─────────────────────────
                st.markdown('<div class="section-title">Forecast Values</div>', unsafe_allow_html=True)
                disp_df = forecast_df.copy()
                disp_df["vs. Last Close"] = (
                    (disp_df["Predicted_Close"] - last_price) / last_price * 100
                ).round(2).astype(str) + "%"
                disp_df["Direction"] = disp_df["Predicted_Close"].apply(
                    lambda p: "▲" if p >= last_price else "▼"
                )
                st.dataframe(disp_df, use_container_width=True)

                # ── Model test metrics ──────────────────────
                if "test_metrics" in meta:
                    st.markdown('<div class="section-title" style="margin-top:1.5rem">Model Performance (Test Set)</div>',
                                unsafe_allow_html=True)
                    cols = st.columns(len(meta["test_metrics"]))
                    for i, (k, v) in enumerate(meta["test_metrics"].items()):
                        with cols[i]:
                            st.markdown(f"""<div class="metric-card">
                                <div class="metric-label">{k}</div>
                                <div class="metric-value" style="font-size:1.2rem">{v}</div>
                                </div>""", unsafe_allow_html=True)

            except FileNotFoundError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Forecast error: {e}")
                st.exception(e)
        else:
            st.markdown(
                '<div class="info-box">Click <b>Run Forecast</b> or press the sidebar button to generate predictions.</div>',
                unsafe_allow_html=True
            )

# ═══════════════════════════════════════════════════════════════
# TAB 4 — Model Analysis
# ═══════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">Model Analysis & Insights</div>', unsafe_allow_html=True)

    hist_df = load_training_history(MODEL_DIR)
    meta_path = os.path.join(MODEL_DIR, "metadata.json")

    if hist_df is not None and os.path.exists(meta_path):
        with open(meta_path) as f:
            meta_data = json.load(f)

        # Training curves
        col_a, col_b = st.columns(2)
        with col_a:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=hist_df["loss"], name="Train Loss",
                                     line=dict(color="#00d4ff", width=2)))
            if "val_loss" in hist_df.columns:
                fig.add_trace(go.Scatter(y=hist_df["val_loss"], name="Val Loss",
                                         line=dict(color="#7c3aed", width=2)))
            fig.update_layout(**PLOT_THEME, height=280, title="Loss Curves",
                              xaxis_title="Epoch", yaxis_title="Huber Loss")
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            if "mae" in hist_df.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=hist_df["mae"], name="Train MAE",
                                         line=dict(color="#10b981", width=2)))
                if "val_mae" in hist_df.columns:
                    fig.add_trace(go.Scatter(y=hist_df["val_mae"], name="Val MAE",
                                             line=dict(color="#f59e0b", width=2)))
                fig.update_layout(**PLOT_THEME, height=280, title="MAE Curves",
                                  xaxis_title="Epoch", yaxis_title="MAE")
                st.plotly_chart(fig, use_container_width=True)

        # Metadata summary
        st.markdown('<div class="section-title">Training Summary</div>', unsafe_allow_html=True)
        summary_cols = st.columns(3)
        info_items = [
            ("Trained On", meta_data.get("ticker", "—")),
            ("Epochs Run", meta_data.get("epochs_run", "—")),
            ("Train Samples", meta_data.get("train_samples", "—")),
            ("Val Samples", meta_data.get("val_samples", "—")),
            ("Test Samples", meta_data.get("test_samples", "—")),
            ("Features", meta_data.get("n_features", "—")),
        ]
        for i, (label, val) in enumerate(info_items):
            with summary_cols[i % 3]:
                st.markdown(f"""<div class="metric-card" style="margin-bottom:0.8rem">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value" style="font-size:1.2rem">{val}</div>
                    </div>""", unsafe_allow_html=True)

        if "test_metrics" in meta_data:
            st.markdown('<div class="section-title" style="margin-top:1.5rem">Test Set Performance</div>',
                        unsafe_allow_html=True)
            st.markdown(
                pd.DataFrame([meta_data["test_metrics"]]).to_html(index=False),
                unsafe_allow_html=True
            )
    else:
        st.markdown(
            '<div class="info-box">No training history found. Train a model first to see analysis.</div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Feature Engineering Overview</div>', unsafe_allow_html=True)

    features_info = {
        "OHLCV Base": ["Open", "High", "Low", "Close", "Volume"],
        "Moving Averages": ["MA_7", "MA_21", "MA_50", "EMA_12", "EMA_26"],
        "Momentum": ["MACD", "MACD_Signal", "RSI"],
        "Volatility": ["BB_Upper", "BB_Lower", "BB_Width", "Volatility"],
        "Returns": ["Return_1d", "Return_5d", "Return_21d"],
        "Volume": ["Volume_MA", "Volume_Ratio"],
    }

    cols = st.columns(3)
    for i, (group, feats) in enumerate(features_info.items()):
        with cols[i % 3]:
            tags = "".join([f'<span class="forecast-tag">{f}</span>' for f in feats])
            st.markdown(
                f'<div class="metric-card" style="text-align:left;margin-bottom:0.8rem">'
                f'<div class="metric-label" style="margin-bottom:0.6rem">{group}</div>'
                f'{tags}</div>',
                unsafe_allow_html=True
            )

# ─── Footer ──────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center;font-family:Space Mono,monospace;font-size:0.7rem;color:#374151;">'
    'DeepForecast · LSTM-Transformer Time Series Forecasting · Built with TensorFlow & Streamlit'
    '</div>',
    unsafe_allow_html=True
)
