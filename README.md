# 📈 DeepForecast — Advanced Stock Price Forecasting

> End-to-end time series forecasting system using a **LSTM–Transformer hybrid** architecture with TensorFlow/Keras and a Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)

---

## 🧠 Architecture

```
Input(seq_len, n_features)
       ↓
TCNBlock(64,  dilation=1)   ← Causal dilated convolutions
TCNBlock(64,  dilation=2)   ← Wider receptive field
TCNBlock(128, dilation=4)   ← Global temporal context
       ↓
Dense(128) → LayerNorm      ← Feature projection
       ↓
TransformerBlock × 2        ← Multi-head self-attention
       ↓
BiLSTM(128) → BiLSTM(64)   ← Sequential memory
       ↓
Temporal Attention Pool     ← Focus on important timesteps
       ↓
Dense(256) → Dense(128)     ← Prediction head
       ↓
Dense(forecast_horizon)     ← Multi-step output
```

**Key design choices:**
- **Huber Loss** — robust to price spike outliers
- **Adam + gradient clipping** — stable training
- **Early Stopping + ReduceLROnPlateau** — auto-regularization
- **22 engineered features** — technical indicators beyond raw OHLCV

---

## 🚀 Quickstart

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/deepforecast.git
cd deepforecast
pip install -r requirements.txt
```

### 2. Train a model

```bash
python train.py --ticker AAPL --period 5y --seq_length 60 --horizon 7 --epochs 100
```

**Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--ticker` | `AAPL` | Yahoo Finance ticker symbol |
| `--period` | `5y` | Historical data period (`2y`, `5y`, `10y`, `max`) |
| `--seq_length` | `60` | Past days used as input context |
| `--horizon` | `7` | Future days to predict |
| `--epochs` | `100` | Max training epochs |
| `--batch_size` | `32` | Mini-batch size |
| `--lr` | `0.001` | Initial learning rate |
| `--model_dir` | `models/saved` | Where to save model + scalers |

### 3. Launch the Streamlit app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 📊 Features

### Market Data Tab
- Interactive candlestick + volume chart with MA overlays
- RSI and MACD indicators
- 52-week high/low, daily change KPIs

### Train Model Tab
- Launch training directly from the UI
- Live training logs
- Loss/MAE training curves

### Forecast Tab
- Multi-day price forecast with confidence band
- Direction and percentage change predictions
- Full forecast table with vs. last-close comparison

### Model Analysis Tab
- Training/validation loss and MAE curves
- Test set metrics (MAE, RMSE, MAPE, R², Directional Accuracy)
- Feature engineering overview

---

## 📐 Feature Engineering (22 features)

| Group | Features |
|-------|----------|
| Base | Open, High, Low, Close, Volume |
| Moving Averages | MA_7, MA_21, MA_50, EMA_12, EMA_26 |
| Momentum | MACD, MACD_Signal, RSI(14) |
| Volatility | BB_Upper, BB_Lower, BB_Width, Rolling Volatility |
| Returns | Return_1d, Return_5d, Return_21d |
| Volume | Volume_MA, Volume_Ratio |

---

## 📁 Project Structure

```
deepforecast/
├── app.py                  # Streamlit application
├── train.py                # Training pipeline
├── predict.py              # Inference engine
├── requirements.txt
├── models/
│   ├── __init__.py
│   └── forecaster.py       # Model architecture
├── utils/
│   ├── __init__.py
│   ├── data_loader.py      # Data fetching + preprocessing
│   └── metrics.py          # Evaluation metrics
├── models/saved/           # Auto-created on training
│   ├── forecaster.keras
│   ├── metadata.json
│   ├── training_history.csv
│   └── scalers/
└── .streamlit/
    └── config.toml
```
