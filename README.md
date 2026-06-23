# Cross-Sectional Momentum: Replication and ML Extension

Replicating the momentum anomaly (Jegadeesh & Titman, 1993) from scratch using Ken French's data, then extending the analysis with a gradient-boosted signal combination model. The goal is to evaluate whether momentum survives when assessed with practitioner metrics: IC, turnover, realistic transaction costs, and whether a walk-forward ML model adds anything on top of the linear factor.

---

## Motivation

Momentum is one of the most studied anomalies in empirical finance and sits at the core of systematic equity strategies at firms like AQR. Most publicly available replications stop at a Sharpe ratio comparison. This project tries to go further: treat it the way a practitioner would, with attention to construction choices, time variation, and eventually non-linear signal combination.

---

## Structure

```
ff-momentum/
├── 01_data_acquisition.py       # French data download, cleaning, exploratory charts
├── 02_factor_construction.py    # Build momentum signal from individual stock data
├── 03_performance_analysis.py   # IC, ICIR, turnover, transaction cost adjustment
├── 04_ml_extension.py           # XGBoost with walk-forward validation
├── results/                     # All generated figures
├── environment.yml
└── README.md
```

Data files are gitignored. Run scripts in order.

---

## Setup

```bash
conda env create -f environment.yml
conda activate ffmomentum
python 01_data_acquisition.py
```

---

## Data

- **Ken French Data Library**: 10 momentum-sorted portfolios and FF3 factors, monthly from 1927. Value-weighted returns throughout.
- **Individual stock data** (scripts 02+): pulled via yfinance. Survivorship bias applies since the universe is current S&P 500 constituents. This is a known limitation acknowledged in the analysis.

---

## Findings *(updated as project develops)*

- WML earns roughly X% annualised over the full sample, but with severe left tail risk-- the 2008–09 crash alone produced a drawdown of ~Y%
- Rolling Sharpe shows significant time variation; the post-2000 period looks weaker, which raises questions about crowding and capacity
- *(ML results pending)*

---

## A few methodological notes

**Why skip the most recent month?**
One-month return reversal is well documented and would contaminate a momentum signal formed on t-1. The standard formation window is t-12 to t-2.

**Why value-weighted portfolios?**
Equal-weighting loads heavily on micro-caps, which are expensive to trade and inflate paper returns. Value-weighting gives a more realistic picture of what's actually capturable.

**Walk-forward validation**
Standard k-fold CV leaks future data into training when applied to time series. Every model in script 04 is trained only on data available at the point of prediction.

---

## References

- Jegadeesh, N. & Titman, S. (1993). Returns to buying winners and selling losers. *Journal of Finance*.
- Asness, C., Moskowitz, T. & Pedersen, L. (2013). Value and momentum everywhere. *Journal of Finance*.
- French, K. Data Library — https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
