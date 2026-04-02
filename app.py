import sys
import warnings
import math
import datetime
import os

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

# ── yfinance import — show clear fix if still on old version ──────
try:
    import yfinance as yf
    _yf_ver = tuple(int(x) for x in yf.__version__.split(".")[:3])
    if _yf_ver < (0, 2, 40):
        st.warning(
            f"⚠️ You have yfinance **{yf.__version__}** — this version is too old and "
            "will fail with 'JSONDecodeError' or 'No timezone found'.\n\n"
            "Run this in your terminal, then restart Streamlit:\n"
            "```\npip uninstall yfinance multitasking -y\n"
            'pip install "yfinance>=0.2.40"\n```'
        )
except TypeError:
    st.error(
        "yfinance import failed (Python 3.8 + old multitasking conflict).\n\n"
        "Fix:\n```\npip uninstall yfinance multitasking -y\n"
        'pip install "yfinance>=0.2.40"\n```'
    )
    st.stop()
except Exception as e:
    st.error(f"Cannot import yfinance: {e}")
    st.stop()

# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="LSTM Stock Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #050a0f; }
  [data-testid="stSidebar"]          { background: #080f18; border-right:1px solid #1a3050; }
  .stButton > button {
    background:#00e5ff15; border:1px solid #00e5ff;
    color:#00e5ff; font-family:monospace; letter-spacing:.06em; transition:all .2s;
  }
  .stButton > button:hover { background:#00e5ff30; }
  h1,h2,h3 { color:#c8dff0 !important; }
  .stProgress > div > div { background:linear-gradient(90deg,#00e5ff,#00ff9d) !important; }
  code { color:#00e5ff !important; background:#0a1520 !important; }
  [data-testid="stMarkdownContainer"] p { color:#c8dff0; }
  div[data-testid="metric-container"] { background:#0a1520; border:1px solid #0f2035;
                                        padding:.75rem; border-radius:4px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# TICKER CATALOGUE — dropdown instead of free-text input
# ══════════════════════════════════════════════════════════════════
TICKERS = {
    # ── US Index ETFs ──────────────────────────────────────────────
    "SPY  — S&P 500 ETF (top 500 US companies)":               "SPY",
    "QQQ  — NASDAQ 100 ETF (tech-heavy)":                      "QQQ",
    "DIA  — Dow Jones Industrial Average ETF":                 "DIA",
    "IWM  — Russell 2000 Small-Cap ETF":                       "IWM",
    "VTI  — Total US Stock Market ETF":                        "VTI",
    "VOO  — Vanguard S&P 500 ETF":                             "VOO",
    # ── Sector ETFs ────────────────────────────────────────────────
    "XLK  — Technology Sector ETF":                            "XLK",
    "XLF  — Financial Sector ETF":                             "XLF",
    "XLE  — Energy Sector ETF":                                "XLE",
    "XLV  — Healthcare Sector ETF":                            "XLV",
    "XLY  — Consumer Discretionary ETF":                       "XLY",
    # ── Big Tech Stocks ────────────────────────────────────────────
    "AAPL — Apple Inc.":                                       "AAPL",
    "MSFT — Microsoft Corporation":                            "MSFT",
    "GOOGL— Alphabet (Google)":                                "GOOGL",
    "AMZN — Amazon.com Inc.":                                  "AMZN",
    "NVDA — NVIDIA Corporation":                               "NVDA",
    "META — Meta Platforms (Facebook)":                        "META",
    "TSLA — Tesla Inc. (high volatility)":                     "TSLA",
    # ── Other Assets ───────────────────────────────────────────────
    "GLD  — Gold ETF":                                         "GLD",
    "TLT  — US Treasury 20yr Bond ETF":                        "TLT",
    "BTC-USD — Bitcoin (USD)":                                 "BTC-USD",
}

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⟨ LSTM Fund Analyzer ⟩")
    st.markdown("---")

    # Dropdown instead of text input
    ticker_label = st.selectbox(
        "Select Asset",
        options=list(TICKERS.keys()),
        index=0,
        help="Choose a stock or ETF to analyze"
    )
    ticker = TICKERS[ticker_label]
    st.caption(f"Ticker symbol: **`{ticker}`**")

    period_label = st.selectbox(
        "Historical Data Period",
        options=["1 Year", "2 Years", "5 Years", "10 Years", "Max Available"],
        index=2,
        help="More data = better LSTM training. Recommended: 5 Years minimum."
    )
    period_map = {
        "1 Year": 1, "2 Years": 2, "5 Years": 5,
        "10 Years": 10, "Max Available": 20
    }
    period_years = period_map[period_label]

    window_size = st.slider(
        "Lookback Window (trading days)",
        min_value=20, max_value=120, value=60, step=5,
        help="How many past days the LSTM sees before predicting"
    )
    forecast_days = st.slider(
        "Forecast Horizon (days ahead)",
        min_value=5, max_value=30, value=10, step=5,
        help="How many days into the future the model predicts"
    )
    return_threshold = st.slider(
        "BUY Signal Threshold (%)",
        min_value=0.5, max_value=5.0, value=1.5, step=0.5,
        help="Min expected % gain to classify a day as BUY"
    ) / 100.0

    st.markdown("---")
    st.markdown("**Model Architecture**")
    lstm_units_1 = st.select_slider("LSTM Layer 1 Units", [32, 64, 128, 256], value=128)
    lstm_units_2 = st.select_slider("LSTM Layer 2 Units", [16, 32, 64, 128], value=64)
    dropout_rate = st.slider("Dropout Rate", 0.1, 0.5, 0.3, 0.05)
    epochs       = st.slider("Max Training Epochs", 20, 150, 80, 10)

    st.markdown("---")
    run_btn = st.button("🚀  Run Analysis", use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# DATA LOADING — robust, handles yfinance API changes
# ══════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_data(ticker: str, years: int) -> pd.DataFrame:
    """
    Download OHLCV data using explicit start/end dates.

    Uses Ticker().history() — the most stable method in yfinance >= 0.2.40.
    Strips timezone from the DatetimeIndex so downstream pandas operations
    (comparisons, merges) work without tz-aware vs tz-naive errors.
    """
    end_dt   = datetime.date.today()
    start_dt = end_dt - datetime.timedelta(days=365 * years)

    tk = yf.Ticker(ticker)
    df = tk.history(
        start=start_dt.strftime("%Y-%m-%d"),
        end=end_dt.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,   # adjusts for splits and dividends automatically
        actions=False       # we don't need dividend/split columns
    )

    # Flatten MultiIndex columns if present (some yfinance versions return them)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Strip timezone — yfinance returns tz-aware index, pandas operations
    # can fail when mixing tz-aware and tz-naive timestamps later
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Keep only the OHLCV columns we need
    needed = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[needed].dropna()
    return df


# ══════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build 17 candlestick + technical indicator features.
    No look-ahead bias: every value only uses data available on or before that day.
    """
    f = pd.DataFrame(index=df.index)

    # ── Candlestick body features ──────────────────────────────────
    f["body"]         = df["Close"] - df["Open"]
    f["body_ratio"]   = f["body"].abs() / (df["High"] - df["Low"] + 1e-8)
    f["upper_shadow"] = df["High"] - df[["Open", "Close"]].max(axis=1)
    f["lower_shadow"] = df[["Open", "Close"]].min(axis=1) - df["Low"]
    f["direction"]    = np.sign(f["body"])

    # ── Return features ────────────────────────────────────────────
    f["return_1d"]  = df["Close"].pct_change(1)
    f["return_5d"]  = df["Close"].pct_change(5)
    f["return_10d"] = df["Close"].pct_change(10)

    # ── Trend / moving average features ───────────────────────────
    sma20 = df["Close"].rolling(20).mean()
    sma50 = df["Close"].rolling(50).mean()
    f["price_vs_sma20"] = df["Close"] / (sma20 + 1e-8)
    f["price_vs_sma50"] = df["Close"] / (sma50 + 1e-8)
    f["sma_cross"]      = sma20 / (sma50 + 1e-8)

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    f["macd"]           = (ema12 - ema26) / (df["Close"] + 1e-8)

    # ── Volatility features ────────────────────────────────────────
    f["volatility_20d"] = f["return_1d"].rolling(20).std()
    f["atr_ratio"]      = (df["High"] - df["Low"]).rolling(14).mean() / (df["Close"] + 1e-8)

    # ── Volume features ────────────────────────────────────────────
    vol_avg20          = df["Volume"].rolling(20).mean()
    f["volume_ratio"]  = df["Volume"] / (vol_avg20 + 1)
    obv                = (np.sign(f["return_1d"]) * df["Volume"]).cumsum()
    f["obv_ratio"]     = obv / (obv.abs().rolling(20).mean() + 1)

    # ── RSI (Relative Strength Index) — normalized to [0,1] ────────
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    f["rsi"] = (100 - 100 / (1 + gain / (loss + 1e-8))) / 100.0

    return f.dropna()


# ══════════════════════════════════════════════════════════════════
# LABELS, SEQUENCES, MODEL
# ══════════════════════════════════════════════════════════════════

def create_labels(df_orig, feat_index, horizon, threshold):
    aligned = df_orig.loc[feat_index, "Close"]
    ret     = aligned.shift(-horizon) / aligned - 1
    return (ret > threshold).astype(int)


def make_sequences(features_arr, labels_arr, window):
    X, y = [], []
    for i in range(window, len(features_arr)):
        X.append(features_arr[i - window: i])
        y.append(labels_arr[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def build_lstm_model(window, n_features, units1, units2, drop):
    """
    2-layer stacked LSTM classifier using tf.keras built-in layers only.
    Input  shape: (batch, window, n_features)
    Output shape: (batch, 1) — sigmoid probability of BUY
    """
    mdl = models.Sequential([
        layers.LSTM(units1, return_sequences=True,
                    input_shape=(window, n_features), name="lstm_1"),
        layers.Dropout(drop, name="drop_1"),
        layers.LSTM(units2, return_sequences=False, name="lstm_2"),
        layers.Dropout(drop, name="drop_2"),
        layers.Dense(32, activation="relu", name="dense"),
        layers.Dropout(drop / 2, name="drop_3"),
        layers.Dense(1, activation="sigmoid", name="output")
    ], name="LSTM_Fund_Analyzer")
    mdl.compile(
        optimizer=tf.keras.optimizers.Adam(0.0005),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return mdl


def decode_signal(prob):
    if prob >= 0.65:   return "STRONG BUY",  "buy",  "🟢"
    elif prob >= 0.52: return "BUY",          "buy",  "🟩"
    elif prob >= 0.35: return "HOLD / WAIT",  "hold", "🟡"
    elif prob >= 0.20: return "SELL",         "sell", "🟥"
    else:              return "STRONG SELL",  "sell", "🔴"


# ══════════════════════════════════════════════════════════════════
# CHART HELPERS
# ══════════════════════════════════════════════════════════════════

def candlestick_fig(df, ticker):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing_line_color="#00ff9d", decreasing_line_color="#ff4d4d",
        name=ticker), row=1, col=1)
    colors = ["#00ff9d" if c >= o else "#ff4d4d"
              for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"],
                         marker_color=colors, opacity=0.5, name="Volume"),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"].rolling(20).mean(),
                             line=dict(color="#00e5ff", width=1.2), name="SMA 20"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"].rolling(50).mean(),
                             line=dict(color="#b06cff", width=1.2), name="SMA 50"),
                  row=1, col=1)
    fig.update_layout(plot_bgcolor="#050a0f", paper_bgcolor="#050a0f",
                      font_color="#c8dff0", height=520,
                      xaxis_rangeslider_visible=False,
                      legend=dict(bgcolor="#0a1520", bordercolor="#1a3050", borderwidth=1),
                      margin=dict(l=10, r=10, t=30, b=10))
    fig.update_xaxes(gridcolor="#0f2035"); fig.update_yaxes(gridcolor="#0f2035")
    return fig


def loss_fig(history):
    ep  = list(range(1, len(history.history["loss"]) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ep, y=history.history["loss"],
                             name="Train Loss", line=dict(color="#00e5ff", width=2)))
    fig.add_trace(go.Scatter(x=ep, y=history.history["val_loss"],
                             name="Val Loss",
                             line=dict(color="#ffb800", width=2, dash="dash")))
    fig.update_layout(plot_bgcolor="#050a0f", paper_bgcolor="#050a0f",
                      font_color="#c8dff0", height=300,
                      xaxis_title="Epoch", yaxis_title="Loss",
                      legend=dict(bgcolor="#0a1520"),
                      margin=dict(l=10, r=10, t=20, b=10))
    fig.update_xaxes(gridcolor="#0f2035"); fig.update_yaxes(gridcolor="#0f2035")
    return fig


def signal_history_fig(dates, probs):
    fig = go.Figure()
    fig.add_hrect(y0=0.65, y1=1.0, fillcolor="#00ff9d", opacity=0.07,
                  annotation_text="BUY zone", annotation_font_color="#00ff9d")
    fig.add_hrect(y0=0.35, y1=0.65, fillcolor="#ffb800", opacity=0.05,
                  annotation_text="HOLD zone", annotation_font_color="#ffb800")
    fig.add_hrect(y0=0.0,  y1=0.35, fillcolor="#ff4d4d", opacity=0.07,
                  annotation_text="SELL zone", annotation_font_color="#ff4d4d")
    fig.add_trace(go.Scatter(x=dates, y=probs, mode="lines",
                             name="BUY Probability",
                             line=dict(color="#00e5ff", width=2),
                             fill="tozeroy", fillcolor="rgba(0,229,255,0.05)"))
    fig.add_hline(y=0.5, line_dash="dot", line_color="#ffffff", opacity=0.3)
    fig.update_layout(plot_bgcolor="#050a0f", paper_bgcolor="#050a0f",
                      font_color="#c8dff0", height=280,
                      yaxis=dict(range=[0, 1], title="BUY Probability"),
                      margin=dict(l=10, r=10, t=20, b=10))
    fig.update_xaxes(gridcolor="#0f2035"); fig.update_yaxes(gridcolor="#0f2035")
    return fig


# ══════════════════════════════════════════════════════════════════
# LANDING PAGE (before Run is clicked)
# ══════════════════════════════════════════════════════════════════

st.title("📈 LSTM Stock Fund Investment Analyzer")
st.caption(
    "Uses `tf.keras.layers.LSTM` (TensorFlow built-in) to detect candlestick patterns "
    "and generate **BUY / HOLD / SELL** signals with confidence scores."
)
st.markdown("---")

if not run_btn:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### How it works")
        st.markdown("""
1. Downloads real OHLCV data via Yahoo Finance
2. Engineers 17 candlestick + technical features
3. Trains a 2-layer stacked LSTM classifier
4. Outputs BUY / HOLD / SELL with confidence %
        """)
    with c2:
        st.markdown("#### LSTM Architecture")
        st.code("""LSTM(128, return_sequences=True)
Dropout(0.3)
LSTM(64,  return_sequences=False)
Dropout(0.3)
Dense(32, activation='relu')
Dense(1,  activation='sigmoid')""", language="python")
    with c3:
        st.markdown("#### Required install")
        st.code("""pip uninstall yfinance multitasking -y
pip install "yfinance>=0.2.40"
pip install streamlit plotly
pip install scikit-learn tensorflow""", language="bash")

    st.info("👈 Select an asset from the sidebar and click **Run Analysis**")
    st.stop()


# ══════════════════════════════════════════════════════════════════
# PIPELINE EXECUTION
# ══════════════════════════════════════════════════════════════════

bar = st.progress(0, text="")

# ── STEP 1: Download ──────────────────────────────────────────────
bar.progress(5, text=f"📥 Downloading {ticker} ({period_label})...")
try:
    df = load_data(ticker, period_years)
except Exception as e:
    st.error(f"Download failed for **{ticker}**: {e}\n\n"
             "Make sure you have run:\n"
             "```\npip uninstall yfinance multitasking -y\n"
             'pip install "yfinance>=0.2.40"\n```')
    st.stop()

if df is None or df.empty:
    st.error(
        f"**`{ticker}` returned empty data.**\n\n"
        "Most likely cause: yfinance version is too old.\n\n"
        "Fix (run in terminal, then restart Streamlit):\n"
        "```\npip uninstall yfinance multitasking -y\n"
        'pip install "yfinance>=0.2.40"\n```'
    )
    st.stop()

if len(df) < 200:
    st.warning(f"Only **{len(df)} rows** downloaded. Try a longer period for better results.")

# ── STEP 2: Price chart ───────────────────────────────────────────
bar.progress(15, text="📊 Rendering chart...")
st.subheader(f"📊 {ticker}  —  {period_label} Price History")
st.plotly_chart(candlestick_fig(df, ticker), use_container_width=True)

last_close = float(df["Close"].iloc[-1])
prev_close = float(df["Close"].iloc[-2])
chg        = (last_close - prev_close) / prev_close * 100
hi52       = float(df["Close"].tail(252).max())
lo52       = float(df["Close"].tail(252).min())

co1, co2, co3, co4 = st.columns(4)
co1.metric("Last Close",   f"${last_close:.2f}", f"{chg:+.2f}%")
co2.metric("52-Week High", f"${hi52:.2f}")
co3.metric("52-Week Low",  f"${lo52:.2f}")
co4.metric("Trading Days", f"{len(df):,}")
st.markdown("---")

# ── STEP 3: Feature engineering ───────────────────────────────────
bar.progress(25, text="⚙️  Engineering features...")
features_df  = engineer_features(df)
feature_cols = features_df.columns.tolist()
n_features   = len(feature_cols)

with st.expander(f"🔬 Engineered features ({n_features} columns) — last 10 rows"):
    st.dataframe(features_df.tail(10).style.format("{:.4f}"),
                 use_container_width=True)

# ── STEP 4: Labels ────────────────────────────────────────────────
bar.progress(30, text="🏷️  Creating BUY/SELL labels...")
labels           = create_labels(df, features_df.index, forecast_days, return_threshold)
labels           = labels.iloc[:-forecast_days]
features_trimmed = features_df.iloc[:-forecast_days]

l1, l2 = st.columns(2)
l1.metric("BUY  labels",  f"{labels.sum():,} ({labels.mean()*100:.1f}%)")
l2.metric("SELL labels",  f"{(labels==0).sum():,} ({(1-labels.mean())*100:.1f}%)")

# ── STEP 5: Scale ─────────────────────────────────────────────────
bar.progress(35, text="📐 Scaling features...")
split_train = int(len(features_trimmed) * 0.80)
scaler      = MinMaxScaler()
scaler.fit(features_trimmed.values[:split_train])
features_scaled = scaler.transform(features_trimmed.values)

# ── STEP 6: Sequences ─────────────────────────────────────────────
bar.progress(40, text="🪟 Creating sliding window sequences...")
X, y   = make_sequences(features_scaled, labels.values, window_size)
trn_n  = int(len(X) * 0.80)
val_n  = int(len(X) * 0.10)

X_train, y_train = X[:trn_n],              y[:trn_n]
X_val,   y_val   = X[trn_n:trn_n+val_n],   y[trn_n:trn_n+val_n]
X_test,  y_test  = X[trn_n+val_n:],        y[trn_n+val_n:]
test_dates        = features_trimmed.index[window_size + trn_n + val_n:]

st.markdown("**Chronological data split (no shuffling):**")
s1, s2, s3 = st.columns(3)
s1.metric("Train", f"{len(X_train):,}")
s2.metric("Val",   f"{len(X_val):,}")
s3.metric("Test",  f"{len(X_test):,}")
st.caption(f"LSTM input shape per sample: `({window_size} timesteps × {n_features} features)`")

# ── STEP 7: Build model ───────────────────────────────────────────
bar.progress(50, text="🏗️  Building LSTM model...")
model = build_lstm_model(window_size, n_features,
                         lstm_units_1, lstm_units_2, dropout_rate)

with st.expander("🏗️ Model Architecture"):
    lines = []
    model.summary(print_fn=lambda x: lines.append(x))
    st.code("\n".join(lines), language="text")

# ── STEP 8: Train ─────────────────────────────────────────────────
bar.progress(55, text="🔁 Training LSTM — please wait...")

history = model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=[
        callbacks.EarlyStopping(monitor="val_loss", patience=12,
                                restore_best_weights=True, verbose=0),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                   patience=6, min_lr=1e-6, verbose=0)
    ],
    verbose=0
)
actual_epochs = len(history.history["loss"])

# ── STEP 9: Evaluate ──────────────────────────────────────────────
bar.progress(85, text="📈 Evaluating...")
y_prob = model.predict(X_test, verbose=0).flatten()
y_pred = (y_prob > 0.5).astype(int)
acc    = accuracy_score(y_test, y_pred)
try:
    auc = roc_auc_score(y_test, y_prob)
except Exception:
    auc = float("nan")

# ── STEP 10: Latest signal ────────────────────────────────────────
bar.progress(95, text="🔮 Generating signal...")
latest_prob   = float(model.predict(
    features_scaled[-window_size:].reshape(1, window_size, n_features),
    verbose=0)[0][0])
signal_label, signal_type, signal_icon = decode_signal(latest_prob)

bar.progress(100, text="✅ Done!")
bar.empty()

# ══════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════

st.markdown("---")
st.subheader("🔮 Investment Decision")

sig_color = {"buy": "#00ff9d", "hold": "#ffb800", "sell": "#ff4d4d"}[signal_type]
st.markdown(f"""
<div style="background:#0a1520;border:2px solid {sig_color};padding:2rem;
            text-align:center;border-radius:4px;margin-bottom:1.5rem;">
  <div style="font-size:3.5rem;font-weight:900;color:{sig_color};font-family:monospace;">
    {signal_icon} {signal_label}
  </div>
  <div style="color:#7a9ab5;font-family:monospace;font-size:.85rem;
              letter-spacing:.1em;margin-top:.5rem;">
    MODEL BUY PROBABILITY: {latest_prob*100:.1f}%
    &nbsp;|&nbsp; TICKER: {ticker}
    &nbsp;|&nbsp; FORECAST: {forecast_days} TRADING DAYS
  </div>
</div>
""", unsafe_allow_html=True)

advice = {
    "buy": f"""
**Why BUY?** The LSTM detected a **bullish candlestick pattern** over the last {window_size} trading days of **{ticker}**.
The model gives a **{latest_prob*100:.1f}% probability** that the price will rise more than {return_threshold*100:.1f}%
in the next {forecast_days} trading days.

**Suggested action:** Consider entering a position. Place a stop-loss ~2% below entry price. Watch for confirmation
via increasing volume on up-candles.
""",
    "hold": f"""
**Why HOLD?** The LSTM sees a **mixed, sideways pattern** in **{ticker}**. BUY probability is {latest_prob*100:.1f}% —
not strong enough to commit capital, not weak enough to short/exit.

**Suggested action:** Stay on the sidelines. Wait for a cleaner breakout in either direction. Re-run after the
next major earnings release or economic event.
""",
    "sell": f"""
**Why SELL / AVOID?** The LSTM detected a **bearish pattern** in **{ticker}**. BUY probability is only {latest_prob*100:.1f}%,
meaning the model leans {(1-latest_prob)*100:.1f}% bearish for the next {forecast_days} trading days.

**Suggested action:** Avoid new positions. If already holding, consider tightening your stop-loss. Wait for a
confirmed bullish reversal (e.g., hammer candle + volume spike) before re-entering.
"""
}[signal_type]

st.markdown(advice)
st.warning("⚠️ **Disclaimer:** For educational purposes only. Never make real investment decisions based solely on "
           "ML signals. Always consult a certified financial advisor.")

st.markdown("---")
st.subheader("📊 Model Performance on Held-Out Test Data")

p1, p2, p3, p4 = st.columns(4)
p1.metric("Test Accuracy",  f"{acc*100:.1f}%")
p2.metric("ROC-AUC",        f"{auc:.3f}" if not math.isnan(auc) else "N/A")
p3.metric("Epochs Trained", str(actual_epochs))
p4.metric("Test Samples",   f"{len(X_test):,}")

st.caption("**Accuracy** = % of days correctly predicted.  "
           "**ROC-AUC** above 0.55 is meaningful for financial data (0.5 = random, 1.0 = perfect).")

with st.expander("📋 Full Classification Report"):
    rpt = classification_report(y_test, y_pred,
                                target_names=["SELL/HOLD", "BUY"],
                                output_dict=True)
    st.dataframe(pd.DataFrame(rpt).T.style.format("{:.3f}"),
                 use_container_width=True)

st.markdown("---")

col_l, col_r = st.columns(2)
with col_l:
    st.subheader("📉 Training Loss Curve")
    st.plotly_chart(loss_fig(history), use_container_width=True)
    st.caption("Validation loss rising while train loss falls = overfitting → increase Dropout.")
with col_r:
    st.subheader("📡 BUY Probability — Test Period")
    st.plotly_chart(signal_history_fig(test_dates[:len(y_prob)], y_prob),
                    use_container_width=True)
    st.caption("Green zone >0.65 = BUY. Yellow 0.35–0.65 = HOLD. Red <0.35 = SELL.")

# ── Signals overlaid on price ─────────────────────────────────────
st.markdown("---")
st.subheader("🕯️ BUY / SELL Signals on Price Chart")
test_close = df["Close"].reindex(test_dates[:len(y_prob)])
test_close = test_close.dropna()

if not test_close.empty:
    buy_idx  = test_close.index[y_pred[:len(test_close)] == 1]
    sell_idx = test_close.index[y_pred[:len(test_close)] == 0]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=test_close.index, y=test_close.values,
                              mode="lines", name="Close",
                              line=dict(color="#c8dff0", width=1.5)))
    fig2.add_trace(go.Scatter(x=buy_idx, y=test_close.loc[buy_idx].values,
                              mode="markers", name="BUY",
                              marker=dict(color="#00ff9d", size=7, symbol="triangle-up")))
    fig2.add_trace(go.Scatter(x=sell_idx, y=test_close.loc[sell_idx].values,
                              mode="markers", name="SELL",
                              marker=dict(color="#ff4d4d", size=7, symbol="triangle-down")))
    fig2.update_layout(plot_bgcolor="#050a0f", paper_bgcolor="#050a0f",
                       font_color="#c8dff0", height=380,
                       legend=dict(bgcolor="#0a1520"),
                       margin=dict(l=10, r=10, t=20, b=10))
    fig2.update_xaxes(gridcolor="#0f2035"); fig2.update_yaxes(gridcolor="#0f2035")
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("🔺 Green = model said BUY on that day.  🔻 Red = model said SELL/HOLD.")

st.markdown("---")
st.subheader("🔧 Tuning Guide")
st.markdown("""
| Parameter | Current | Effect if increased |
|---|---|---|
| Lookback Window | """ + str(window_size) + """ days | More historical context; slower training |
| LSTM Layer 1 Units | """ + str(lstm_units_1) + """ | More capacity to learn patterns; risk of overfitting |
| LSTM Layer 2 Units | """ + str(lstm_units_2) + """ | Same as above |
| Dropout Rate | """ + str(dropout_rate) + """ | More regularization; use 0.3–0.4 if overfitting |
| Forecast Horizon | """ + str(forecast_days) + """ days | Longer = harder to predict but more strategic |
| BUY Threshold | """ + f"{return_threshold*100:.1f}%" + """ | Higher = fewer but higher-conviction BUY signals |
""")

st.markdown("---")
st.caption("Built with `tf.keras.layers.LSTM` · `tf.keras.layers.GRU` · "
           "`yfinance` · `Streamlit` · `Plotly`")