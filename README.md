# 📈 Deep Reinforcement Learning for Portfolio Management  
### *(Indian Equity Markets)*

This project implements a **deep reinforcement learning–based portfolio management system** for Indian equity markets (**NIFTY 50 stocks**).  
The agent learns **dynamic asset allocation** by optimizing portfolio returns while accounting for **transaction costs, market dynamics, and risk**.

The approach is inspired by **financial reinforcement learning research** and adapts it to **real Indian stock market data**.

---

## 🚀 Key Features

- Multi-asset portfolio optimization using **Reinforcement Learning**
- **Convolutional Neural Network (CNN)**–based policy network
- **Actor–Critic architecture**
- **Portfolio Vector Memory (PVM)** for stable learning
- **Transaction costs & interest rate modeling**
- Technical indicators as state features:
  - MACD
  - RSI
  - CCI
  - ADX
- **Train / Validation / Test** split on historical data
- Support for **multiple agents**
- Performance evaluation with:
  - Portfolio value evolution
  - Final portfolio weights
  - Maximum drawdown

---

## 📊 Dataset

- **Market:** NSE (India)
- **Stocks:** Selected NIFTY 50 constituents
- **Source:** Yahoo Finance (`yfinance`)
- **Time Period:** `2015-03-11` → `2024-09-05`
- **Frequency:** Daily OHLC data

Each stock is stored as a CSV file under:

```text
nifty50_data/
