"""
01_data_acquisition.py

Downloads and explores the Fama-French momentum data.
All data from Ken French's public library via pandas_datareader.

Outputs:
    data/momentum_portfolios.parquet
    data/ff3_factors.parquet
    results/01_decile_returns.png
    results/01_cumulative_wml.png
    results/01_rolling_sharpe.png
    results/01_drawdown_wml.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas_datareader.data as web
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')


# settings

START = '1927-01-01'
END   = '2024-12-31'

DATA_DIR    = 'data'
RESULTS_DIR = 'results'

plt.style.use('seaborn-v0_8-whitegrid')

# download

print("Fetching data from Ken French library...")

# 10 portfolios sorted on prior 12-2 month return (value-weighted, key=0)
raw_mom = web.DataReader('10_Portfolios_Prior_12_2', 'famafrench',
                         start=START, end=END)
mom_raw = raw_mom[0]

# FF3 factors + risk free rate
raw_ff3 = web.DataReader('F-F_Research_Data_Factors', 'famafrench',
                         start=START, end=END)
ff3_raw = raw_ff3[0]

print(f"Momentum: {mom_raw.shape} | FF3: {ff3_raw.shape}")


#clean

def clean_french(df, label):
    # French reports returns as percentages; i converted to decimals
    df.index = df.index.to_timestamp(how='end')
    df = df / 100.0
    df.columns = df.columns.str.strip()
    print(f"{label}: {df.index[0].date()} to {df.index[-1].date()}, {len(df)} obs")
    return df

mom = clean_french(mom_raw, "Momentum")
ff3 = clean_french(ff3_raw, "FF3")

# renaming deciles
mom.columns = [f'D{i}' for i in range(1, 11)]

# WML: long winners (D10), short losers (D1)
# skip most recent month in formation to avoid shortterm reversal contaminating the signal
mom['WML'] = mom['D10'] - mom['D1']

print(f"\nWML descriptive stats (monthly):")
print(mom['WML'].describe().round(4))

#align and save

combined = pd.concat([mom, ff3], axis=1, join='inner')
print(f"\nCombined: {combined.shape}, {combined.index[0].date()} to {combined.index[-1].date()}")

assert combined.isnull().sum().sum() == 0, "unexpected missing values"

mom.to_parquet(os.path.join(DATA_DIR, 'momentum_portfolios.parquet'))
ff3.to_parquet(os.path.join(DATA_DIR, 'ff3_factors.parquet'))
print("Data saved to /data")

# summary stats

cols = [f'D{i}' for i in range(1, 11)] + ['WML']

stats = pd.DataFrame({
    'Ann. Return (%)' : mom[cols].mean() * 12 * 100,
    'Ann. Vol (%)'    : mom[cols].std() * np.sqrt(12) * 100,
    'Sharpe'          : (mom[cols].mean() * 12) / (mom[cols].std() * np.sqrt(12)),
    'Skew'            : mom[cols].skew(),
    'Max Mo. (%)'     : mom[cols].max() * 100,
    'Min Mo. (%)'     : mom[cols].min() * 100,
}).round(3)

print("\n--- Decile Summary ---")
print(stats.to_string())

#chart 1: average return by decile

fig, ax = plt.subplots(figsize=(12, 6))

decile_cols = [f'D{i}' for i in range(1, 11)]
ann_ret = mom[decile_cols].mean() * 12 * 100

colors = ['#d62728'] * 3 + ['#aec7e8'] * 4 + ['#2ca02c'] * 3

bars = ax.bar(range(1, 11), ann_ret, color=colors,
              edgecolor='white', linewidth=0.8, width=0.7)

for bar, val in zip(bars, ann_ret):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)

wml_ann = mom['WML'].mean() * 12 * 100
ax.text(0.98, 0.05, f'WML spread: {wml_ann:.1f}% / year',
        transform=ax.transAxes, ha='right', fontsize=11,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray'))


ax.set_xlabel('Momentum decile  (1 = biggest losers, 10 = biggest winners)', fontsize=12)
ax.set_ylabel('Annualised return (%)', fontsize=12)
ax.set_title(f'Cross-sectional momentum: average return by decile\n'
             f'US equities, value-weighted | {combined.index[0].year}–{combined.index[-1].year}',
             fontsize=13, fontweight='bold')
ax.set_xticks(range(1, 11))
ax.set_xticklabels([f'D{i}' for i in range(1, 11)])

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, '01_decile_returns.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: 01_decile_returns.png")

# chart 2: cumulative growth of $1

wml_cum = (1 + mom['WML']).cumprod()
mkt_cum = (1 + ff3['Mkt-RF'] + ff3['RF']).cumprod()
d10_cum = (1 + mom['D10']).cumprod()
d1_cum  = (1 + mom['D1']).cumprod()

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(wml_cum.index, wml_cum,  label='WML (long D10, short D1)', color='#2ca02c', linewidth=1.8)
ax.plot(mkt_cum.index, mkt_cum,  label='Market (total return)',     color='#1f77b4', linewidth=1.5, linestyle='--')
ax.plot(d10_cum.index, d10_cum,  label='Winners (D10)',             color='#98df8a', linewidth=1.2, alpha=0.8)
ax.plot(d1_cum.index,  d1_cum,   label='Losers (D1)',               color='#ff9896', linewidth=1.2, alpha=0.8)

# 2008-09 momentum crash — one of the worst on record
ax.axvspan(pd.Timestamp('2008-06-01'), pd.Timestamp('2009-03-01'),
           alpha=0.15, color='red', label='2008–09 crash')

ax.set_yscale('log')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Growth of $1 (log scale)', fontsize=12)
ax.set_title(f'Cumulative performance: WML vs market\n'
             f'US equities | {combined.index[0].year}–{combined.index[-1].year}',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.xaxis.set_major_locator(mdates.YearLocator(10))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, '01_cumulative_wml.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 01_cumulative_wml.png")

# chart 3: rolling 36 month sharpe

WINDOW = 36

wml_excess = mom['WML'] - ff3['RF']
roll_sharpe = (wml_excess.rolling(WINDOW).mean() * 12) / \
              (wml_excess.rolling(WINDOW).std() * np.sqrt(12))

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(roll_sharpe.index, roll_sharpe, color='#9467bd', linewidth=1.5,
        label='36-month rolling Sharpe')
ax.axhline(0,   color='black', linewidth=0.8, linestyle='--', alpha=0.5)
ax.axhline(0.5, color='green', linewidth=0.8, linestyle=':',  alpha=0.7, label='Sharpe = 0.5')
ax.axhline(1.0, color='blue',  linewidth=0.8, linestyle=':',  alpha=0.7, label='Sharpe = 1.0')

ax.fill_between(roll_sharpe.index, roll_sharpe, 0,
                where=(roll_sharpe >= 0), alpha=0.15, color='green', interpolate=True)
ax.fill_between(roll_sharpe.index, roll_sharpe, 0,
                where=(roll_sharpe < 0),  alpha=0.15, color='red',   interpolate=True)

ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Annualised Sharpe', fontsize=12)
ax.set_title(f'WML: rolling 36-month Sharpe ratio\n'
             f'US equities | {combined.index[0].year}–{combined.index[-1].year}',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.xaxis.set_major_locator(mdates.YearLocator(10))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, '01_rolling_sharpe.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 01_rolling_sharpe.png")

#chart 4: drawdown

running_max = wml_cum.cummax()
drawdown    = (wml_cum - running_max) / running_max * 100

worst_date = drawdown.idxmin()
worst_val  = drawdown.min()

fig, ax = plt.subplots(figsize=(12, 6))

ax.fill_between(drawdown.index, drawdown, 0, color='#d62728', alpha=0.6, label='Drawdown')
ax.plot(drawdown.index, drawdown, color='#d62728', linewidth=0.8)

ax.annotate(
    f'Worst: {worst_val:.1f}%\n({worst_date.strftime("%b %Y")})',
    xy=(worst_date, worst_val),
    xytext=(worst_date + pd.DateOffset(years=3), worst_val + 20),
    arrowprops=dict(arrowstyle='->', color='black'),
    fontsize=10,
    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray')
)

ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Drawdown (%)', fontsize=12)
ax.set_title(f'WML: drawdown from peak\n'
             f'US equities | {combined.index[0].year}–{combined.index[-1].year}',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.xaxis.set_major_locator(mdates.YearLocator(10))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(rotation=45)


plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, '01_drawdown_wml.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 01_drawdown_wml.png")

print("\nDone.")
