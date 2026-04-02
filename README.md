---
title: Stock Fund Analyzer
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# 📈 LSTM Stock Fund Investment Analyzer

An AI-powered stock analysis tool that uses a **2-layer stacked LSTM neural network** to analyze candlestick patterns and generate BUY / HOLD / SELL signals with confidence scores.

## What it does
- Downloads real stock & ETF data via Yahoo Finance
- Engineers 17 technical indicators (RSI, MACD, SMA, ATR, OBV and more)
- Trains an LSTM model on historical price patterns
- Predicts whether a stock will go up in the next 5–30 days
- Outputs a clear BUY / HOLD / SELL signal with confidence %

## How to use
1. Select a stock or ETF from the sidebar (SPY, AAPL, NVDA, BTC and more)
2. Choose how much historical data to train on
3. Adjust the model settings if you want
4. Hit **Run Analysis** and wait ~1 minute for training
5. Get your signal!

## Supported Assets
- US Index ETFs (SPY, QQQ, VTI...)
- Sector ETFs (XLK, XLF, XLE...)
- Big Tech stocks (AAPL, MSFT, NVDA, TSLA...)
- Gold, Bonds, Bitcoin

## Tech Stack
- `TensorFlow` — LSTM neural network
- `Streamlit` — web interface
- `Plotly` — interactive charts
- `yfinance` — live market data
- `scikit-learn` — data preprocessing

## Disclaimer
⚠️ For educational purposes only. Never make real investment decisions based solely on ML signals. Always consult a certified financial advisor.
