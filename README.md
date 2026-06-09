# 🚀 Forex Trading Signal Generation System

A complete end-to-end algorithmic trading system that combines **technical analysis + machine learning** to generate, evaluate, and compare trading strategies on EUR/USD data.

---

## 📌 Overview

This project simulates a real-world trading environment by:

* Processing **3 years of hourly EUR/USD data**
* Computing **15+ technical indicators**
* Generating signals using:

  * Rule-based strategy (Composite Signal)
  * Machine Learning models (XGBoost & Random Forest)
* Backtesting strategies with **realistic trading costs**
* Evaluating performance using **20+ financial metrics**
* Exporting results for **Power BI dashboards**

---

## ⚙️ Features

### 📊 Technical Analysis

* RSI (14)
* MACD (12, 26, 9)
* Bollinger Bands (20)
* Moving Averages (SMA, EMA)
* ATR (Volatility)
* Momentum Indicators

---

### 🤖 Machine Learning Models

* **XGBoost Classifier**
* **Random Forest Classifier**

Includes:

* Feature engineering
* Signal classification
* Probability predictions

---

### 📈 Trading Strategies

1. **Composite Signal Strategy**

   * Weighted indicator voting:

     * MA (25%)
     * RSI (20%)
     * MACD (20%)
     * Bollinger Bands (15%)
     * Momentum (20%)

2. **XGBoost Strategy**

   * Data-driven predictions
   * Higher return potential

3. **Random Forest Strategy**

   * More stable predictions
   * Lower variance

---

### 💰 Backtesting Engine

Simulates real trading conditions:

* Commission: `0.0007`
* Slippage: `0.0002`
* Position size: `100,000`
* Trade execution logic
* P&L tracking

---

### 📉 Risk & Performance Metrics

* Win Rate
* Sharpe Ratio
* Profit Factor
* Maximum Drawdown
* Value at Risk (VaR)
* Conditional VaR (CVaR)
* Total Return (%)
* Equity Curve Analysis

---

### 🔁 Monte Carlo Simulation

* 1000 simulated scenarios
* Risk distribution analysis
* Strategy robustness testing

---

### 📊 Power BI Integration

Exported datasets:

* `PowerBI_Signals.csv`
* `PowerBI_Performance_Metrics.csv`
* `PowerBI_Risk_Analysis.csv`
* `PowerBI_Equity_Curves.csv`

---

## 🛠️ Tech Stack

| Category         | Tools Used            |
| ---------------- | --------------------- |
| Programming      | Python                |
| Data Processing  | Pandas, NumPy         |
| Visualization    | Matplotlib, Seaborn   |
| Machine Learning | Scikit-learn, XGBoost |
| Notebook         | Jupyter Notebook      |
| Dashboard        | Power BI              |

---

## 📦 Library Versions

```
Python == 3.10+

pandas == 2.0.3
numpy == 1.24.3
matplotlib == 3.7.1
seaborn == 0.12.2

scikit-learn == 1.3.0
xgboost == 1.7.6

scipy == 1.11.1
```

---

## ▶️ How to Run

1. Clone repository

```
git clone <your-repo-link>
cd forex-trading-system
```

2. Install dependencies

```
pip install -r requirements.txt
```

3. Run notebook

```
jupyter notebook FX_TRADING_SYSTEM_COMPLETE.ipynb
```

4. Execute all cells (top → bottom)

---

## 📊 Outputs

* Trading signals visualization
* ML prediction probabilities
* Strategy comparison charts
* Equity curves
* Risk analysis plots

---

## 🧠 Key Insights

* Machine learning models typically outperform rule-based systems in dynamic markets
* Composite signals provide stability but lower returns
* Risk management (drawdown control) is critical for long-term profitability

---

## ⚠️ Limitations

* Based on historical data (no live trading)
* Assumes constant transaction costs
* Does not include macroeconomic factors or news sentiment

---

## 🔮 Future Improvements

* Add LSTM / Deep Learning models
* Multi-currency support
* Real-time data integration
* Automated trading (API integration)
* Reinforcement learning strategies

---

## 📌 Project Status

✅ Complete
✅ Tested
✅ Ready for Portfolio / Internship Submission

---

## 🎥 Recommended Add-on

Record a short demo video explaining:

* Problem
* Approach
* Results

(This significantly improves recruiter visibility)

---

## 👤 Author

Your Name
LinkedIn: (Add link)
GitHub: (Add link)

---
