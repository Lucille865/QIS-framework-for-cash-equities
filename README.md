# QIS Cash Equities Engine — Volatility Target, Decrement Indexing & Performance Attribution

An institutional-grade Quantitative Investment Strategies (QIS) and index engineering framework in Python. This project simulates end-to-end multi-asset cash equity index mechanics, including currency-hedged excess return calculation, dynamic volatility targeting with volatility adjustment factors (VAF), synthetic decrement mechanisms, and high-performance risk/performance attribution.

---

## 📌 Project Overview

In equity derivatives structuring and structured product manufacturing, underlyings are frequently packaged into rules-based algorithmic indices to optimize option pricing, minimize dividend risk, and smooth risk-adjusted returns:

1. **Excess Return (ER) Structuring & FX Conversion:** Converting gross equity performance into excess return over interbank funding rates (USD 3M Libor/SOFR, EUR Overnight) and cross-currency conversion (EUR/USD).
2. **Dynamic Volatility Targeting (Vol Target):** Dynamically sizing exposure to the equity underlier based on realized volatility to maintain a predefined risk profile (e.g., 15% annualized target volatility), accounting for execution lags, financing/repo drag, and transaction friction.
3. **Decrement Index Engineering:** Applying fixed-point and synthetic percentage dividend decrements to eliminate forward dividend uncertainty for derivative hedgers.
4. **Accelerated Performance Attribution:** Computing vectorized, multi-threaded risk statistics (Sharpe, Sortino, rolling drawdowns) powered by JIT-compiled Numba routines.

---

## 📐 Methodological & Mathematical Framework

### 1. Excess Return (ER) Calculation
Gross total return prices $S_t$ are converted to excess returns by stripping out the cost of funding:
$$ER_t = ER_{t-1} \cdot \left[ 1 + \left( \frac{S_t}{S_{t-1}} - 1 \right) - r_{t-1} \cdot \frac{\text{Act}(t-1, t)}{360} \right]$$
For multi-currency underliers (e.g., US equities packaged for European investors), excess return series are adjusted by the corresponding FX spot rates:
$$ER_t^{\text{EUR}} = ER_t^{\text{USD}} \cdot FX_t^{\text{EUR/USD}}$$

---

### 2. Volatility Target Mechanism (VT)
The strategy targets a constant annualized volatility $\sigma_{\text{target}}$ by dynamically tuning the leverage/exposure $w_t$:

* **Realized Volatility ($HV_t$):**
  $$HV_t = \sqrt{\frac{365}{\text{DayCount}} \cdot \frac{1}{N} \sum_{i=0}^{N-1} \left( \ln \left( \frac{S_{t-i}}{S_{t-i-k}} \right) \right)^2}$$
* **Exposure Sizing ($w_t$):**
  $$w_t = \min \left( w_{\max}, \frac{\sigma_{\text{target}}}{HV_{t-\text{lag}}} \cdot \text{VAF}_{t-\text{lag}} \right)$$
* **Volatility Adjustment Factor ($\text{VAF}_t$):**
  Adapts the target exposure based on the ratio between instantaneous realized volatility ($IHV_t$) and target volatility:
  $$\text{VAF}_t = \text{clip} \left( \sqrt{\max\left(0, 1 + \frac{W_t}{T_{\text{fwd}}} \left(1 - \left(\frac{IHV_t}{\sigma_{\text{target}}}\right)^2\right)\right)}, \text{floor}, \text{cap} \right)$$
* **Index Level Iteration ($IL_t$):**
  $$IL_t = IL_{t-1} \cdot \left[ 1 + w_{t-1} \left(\frac{S_t}{S_{t-1}} - 1 - c_{\text{repo}} \frac{\text{Act}}{360}\right) - c_{\text{struct}} \frac{\text{Act}}{365} - \text{Cost}_{\text{tx}} \right]$$

---

### 3. Decrement Indexing
* **Fixed Point Decrement:**
  $$IL_t^{\text{dec}} = IL_{t-1}^{\text{dec}} \cdot \frac{IL_t}{IL_{t-1}} - D_{\text{pts}} \cdot \frac{\text{Act}(t-1, t)}{365}$$
* **Synthetic Percentage Decrement:**
  $$IL_t^{\text{synth}} = IL_{t-1}^{\text{synth}} \cdot \left( \frac{IL_t}{IL_{t-1}} \right) \cdot \left(1 - \delta_{\text{div}} \cdot \frac{\text{Act}(t-1, t)}{365}\right)$$

---

## ⚡ High-Performance Risk & Performance Attribution

Using parallelized Numba (`@nb.jit(parallel=True)`), the engine benchmarks risk-adjusted performance with low latency:
* **Annualized Return & Volatility** (ACT/365 convention)
* **Sharpe & Sortino Ratios** (downside deviation filtering)
* **Vectorized Rolling Maximum Drawdown (MDD)** over 252-day sliding windows:
  $$\text{MDD}_t = \min_{s \in [t-W, t]} \left( \frac{IL_s}{\max_{u \in [t-W, s]} IL_u} - 1 \right)$$

---

## 🛠️ Tech Stack & Requirements

* **Language:** Python 3.9+ / 3.10+
* **Data Manipulation & Time Series:** `pandas`, `numpy`, `scipy`
* **Performance Acceleration:** `numba` (JIT parallelization), `joblib`
* **Econometrics:** `statsmodels`
* **Visualization:** `matplotlib`

Install dependencies:
```bash
pip install numpy pandas scipy numba statsmodels matplotlib joblib tqdm
```

Clone the repesitory:
```bash
git clone [https://github.com/](https://github.com/)<Lucille865>/<QIS-framework-for-cash-equities>.git
cd <QIS-framework-for-cash-equities>
```

Launch the demonstration notebook:
```bash
jupyter notebook "QIS_Cash_Equities_Demo.ipynb"
```
