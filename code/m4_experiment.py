import json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

DATA_DIR    = Path('/home/claude/forecasting_paper/data')
RESULTS_DIR = Path('/home/claude/forecasting_paper/results')
RESULTS_DIR.mkdir(exist_ok=True)

train_df = pd.read_csv(DATA_DIR / 'm4_monthly_train.csv', index_col=0)
sample   = pd.read_csv(DATA_DIR / 'm4_sample_ids.csv')

results = []
win_counts = {'AR1': 0, 'Naive': 0, 'SeasonalNaive': 0}

for _, row in sample.iterrows():
    mid  = row['id']
    cv_t = row['cv']
    ac1  = row['ac1']
    s    = train_df.loc[mid].dropna().values.astype(float)
    sp   = int(len(s) * 0.8)
    tr, te = s[:sp], s[sp:]

    # Naive mean
    naive_r = float(np.sqrt(mean_squared_error(te, np.full(len(te), tr.mean()))))

    # Seasonal naive (lag-12)
    if len(tr) >= 13:
        sn_preds = [tr[-12 + (i % 12)] for i in range(len(te))]
        sn_r = float(np.sqrt(mean_squared_error(te, sn_preds)))
    else:
        sn_r = naive_r

    # AR(1) with drift
    phi = np.clip(float(pd.Series(tr).autocorr(1)), -0.99, 0.99)
    mu  = tr.mean()
    ar1_preds = np.array([phi * (tr[-1] - mu) + mu] * len(te))
    ar1_r = float(np.sqrt(mean_squared_error(te, ar1_preds)))

    cv_act = s.std() / s.mean()
    best_r = min(naive_r, sn_r, ar1_r)
    if best_r == ar1_r:    winner = 'AR1'
    elif best_r == sn_r:   winner = 'SeasonalNaive'
    else:                  winner = 'Naive'
    win_counts[winner] += 1

    results.append({
        'id': mid, 'cv': round(cv_act, 3), 'ac1': round(ac1, 3), 'n': int(len(s)),
        'arima_rmse': round(ar1_r, 2), 'naive_rmse': round(naive_r, 2),
        'seasonal_naive_rmse': round(sn_r, 2),
        'ratio_vs_naive': round(ar1_r / naive_r, 3), 'winner': winner
    })

# Aggregate by CV bin
df_r = pd.DataFrame(results)
df_r['cv_bin'] = pd.cut(df_r['cv'], [0, 0.1, 0.3, 0.6, 99],
                         labels=['low(<0.1)', 'med(0.1-0.3)', 'high(0.3-0.6)', 'very_high'])

print(f"M4 Micro Monthly — {len(results)} series")
print(f"Win counts: {win_counts}")
print(f"\nMean ratio (AR1/Naive) by CV bin:")
print(df_r.groupby('cv_bin', observed=True)['ratio_vs_naive'].agg(['mean','std','count']).round(3))
print(f"\nOverall mean ratio: {df_r['ratio_vs_naive'].mean():.3f}")
print(f"AR1 beats naive: {(df_r['ratio_vs_naive']<1).sum()}/{len(df_r)}")

with open(RESULTS_DIR / 'm4_micro_results.json', 'w') as f:
    json.dump({
        'results': results,
        'win_counts': win_counts,
        'total': len(results),
        'mean_ratio': round(float(df_r['ratio_vs_naive'].mean()), 3),
        'ar1_beats_naive': int((df_r['ratio_vs_naive'] < 1).sum())
    }, f, indent=2)

print("Saved.")
