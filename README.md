# 📈 CapitalRL  
### Deep Reinforcement Learning for Portfolio Management

**A Self-Designed RL System for Indian Equity Markets**

Not a tutorial clone. Not a Kaggle script.

**CapitalRL** treats portfolio management as a *sequential decision-making problem* and solves it using deep reinforcement learning, realistic market constraints, and real Indian equity data.

This project implements an **end-to-end reinforcement learning pipeline** that learns how to dynamically allocate capital across multiple stocks and cash, optimizing long-term portfolio growth while accounting for transaction costs, market dynamics, and risk.

---

## 💡 Why CapitalRL Exists

Most trading projects focus on predicting prices.  
**Real portfolio management is about allocating capital over time.**

CapitalRL is built around that idea.

### Core Principles

- Decisions are sequential, not independent
- Capital is finite
- Rebalancing has a cost
- Risk matters as much as return

---

## 🚀 What CapitalRL Actually Does

- Learns **how to allocate capital**, not where price will go
- Operates on **multiple stocks + cash**
- Rebalances portfolios while paying **transaction costs**
- Learns directly from market interaction, **not labeled data**

---

## 🧠 Core Features

- **Deep Reinforcement Learning** for portfolio optimization
- **Actor–Critic architecture** with continuous action space
- **CNN-based policy network** for market state encoding
- **Portfolio Vector Memory (PVM)** for stable learning
- Explicit modeling of:
  - Transaction costs
  - Cash allocation
  - Portfolio rebalancing
- **Technical indicators** as part of the state:
  - MACD
  - RSI
  - CCI
  - ADX
- Proper **Train / Validation / Test** split
- Designed to support **multi-agent experimentation**

---

## 📊 Market Data

| Property | Value |
|--------|------|
| **Market** | NSE (India) |
| **Universe** | Selected NIFTY 50 stocks |
| **Source** | Yahoo Finance (yfinance) |
| **Period** | 2015-03-11 → 2024-09-05 |
| **Frequency** | Daily OHLC |

---

## 🧩 State Representation

At every timestep, the agent observes a **3D tensor**:

```
(window_size, num_assets, num_features)
```

### Engineered Features

- Close / Open
- High / Open
- Low / Open
- MACD
- RSI
- CCI
- ADX
- Next-day Open / Current Open

This **price-relative formulation** improves generalization across assets and market regimes.

---

## 🎯 Action Space

**Continuous allocation vector:**

```
[w_cash, w_stock1, w_stock2, ..., w_stockN]
```

- Includes cash
- Softmax-normalized
- Portfolio weights always sum to 1

**No leverage. No unrealistic shortcuts.**

---

## 💰 Reward Design

The agent is rewarded based on **portfolio value evolution**, not raw price changes.

### Reward Function

```
r_t = log(portfolio_value_t / portfolio_value_{t-1})
```

- Encourages long-term growth
- Penalizes drawdowns
- Accounts for transaction costs
- Discourages excessive rebalancing

**Aligned with real investment objectives.**

---

## 🏗️ System Architecture

### Policy Network (Actor)

- CNN feature extractor over market states
- Learnable cash bias
- Softmax output for portfolio allocation

### Value Network (Critic)

- Estimates action-value (Q)
- Consumes:
  - Market state
  - Portfolio weights
  - Portfolio value
  - Action vector

### Portfolio Vector Memory (PVM)

- Stores historical allocations
- Improves training stability
- Reduces variance and oscillations

---

## 🧪 Training Setup

| Component | Technology |
|--------|------------|
| **Frameworks** | TensorFlow, OpenAI Gym |
| **Optimizer** | Adam |
| **Replay Memory** | Experience replay |
| **Training** | Batch-based |
| **Scalability** | tf.distribute.MirroredStrategy |

Built for **clarity, extensibility, and experimentation**.

---

## 📈 Evaluation

Performance is evaluated using:

- Final portfolio value
- Mean / min / max portfolio value
- Maximum drawdown
- Asset allocation dynamics over time

### Visual Analysis

- Portfolio value curve
- Final asset allocation
- Weight evolution per stock

---

## 🛠️ Installation

```bash
pip install tensorflow gym yfinance ta numpy pandas matplotlib seaborn tqdm
```

---

## ▶️ How to Run

1. Download historical market data
2. Preprocess OHLC data and compute indicators
3. Initialize the trading environment
4. Train the RL agent
5. Evaluate on validation/test sets
6. Analyze portfolio behavior visually

---

## 📌 What Makes CapitalRL Different

- Built around **capital allocation**, not price prediction
- Uses **realistic market constraints**
- Research-oriented system design
- Clean separation of environment, agent, and training loop
- Easy to extend to:
  - PPO / SAC
  - Risk-adjusted rewards
  - Larger asset universes
  - Paper trading

---

## 🔮 Next Steps

- Optimize for Sharpe / Sortino ratio
- Risk-aware reward shaping
- PPO / SAC implementation
- Live paper-trading integration
- Hyperparameter tuning
- Multi-market expansion

---

## 👤 Author

**Utkarsh Alshi**  
Software Engineer | Reinforcement Learning | Financial ML  
📍 India

Built out of curiosity, engineering rigour, and a desire to understand how intelligent systems allocate capital under uncertainty.

---
