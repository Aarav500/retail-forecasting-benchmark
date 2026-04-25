"""Generate updated figures with real D-Mart data integrated."""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import sys
sys.path.insert(0, '/home/claude/forecasting_paper/code')
from experiment import *

RESULTS_DIR = Path("/home/claude/forecasting_paper/results")
FIGURES_DIR = Path("/home/claude/forecasting_paper/figures")
DATA_DIR    = Path("/home/claude/forecasting_paper/data")

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.titlesize': 12, 'axes.labelsize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 9, 'figure.dpi': 150,
    'axes.spines.top': False, 'axes.spines.right': False
})
COLORS = {
    'ARIMA':  '#2196F3', 'SARIMA': '#4CAF50', 'Prophet': '#FF9800',
    'XGBoost':'#9C27B0', 'LSTM':   '#F44336', 'Hybrid':  '#795548'
}

def load_all_results():
    r = {}
    for name in ['primary','walmart','m5',
                 'Real_Food','Real_Electronics','Real_Clothing','Real_Furniture']:
        p = RESULTS_DIR / f'{name}_results.json'
        if p.exists():
            with open(p) as f:
                r[name] = json.load(f)
    return r

# ── FIG 1: Master RMSE heatmap (all datasets) ────────────────────────────────
def fig_master_heatmap(results):
    models   = ['ARIMA','SARIMA','Prophet','XGBoost','LSTM','Hybrid']
    datasets = ['Real_Food','Real_Electronics','Real_Clothing','Real_Furniture',
                'walmart','m5']
    labels   = ['Real\nFood','Real\nElec.','Real\nCloth.','Real\nFurn.',
                'Walmart\n(Weekly)','M5\n(Interm.)']

    data = np.zeros((len(models), len(datasets)))
    for j, d in enumerate(datasets):
        for i, m in enumerate(models):
            data[i, j] = results.get(d, {}).get('metrics', {}).get(m, {}).get('rmse', np.nan)

    norm = data / np.nanmax(data, axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(norm, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)));  ax.set_xticklabels(labels)
    ax.set_yticks(range(len(models)));  ax.set_yticklabels(models)

    # Category separator
    ax.axvline(3.5, color='white', lw=3)
    ax.text(1.5, -0.8, 'Real D-Mart Data\n(Low-SNR Regime)',
            ha='center', fontsize=9, color='#37474F', style='italic',
            transform=ax.transData)
    ax.text(4.5, -0.8, 'Public\nBenchmarks',
            ha='center', fontsize=9, color='#37474F', style='italic',
            transform=ax.transData)

    for i in range(len(models)):
        for j in range(len(datasets)):
            v = data[i, j]
            if not np.isnan(v):
                txt = f'{v:.0f}' if v > 100 else f'{v:.2f}'
                ax.text(j, i, txt, ha='center', va='center', fontsize=8,
                        color='black' if norm[i,j] < 0.65 else 'white',
                        fontweight='bold')

    plt.colorbar(im, ax=ax, label='Normalised RMSE', fraction=0.046, pad=0.04)
    ax.set_title('Model Performance Across Demand Regimes\n(Normalised RMSE — lower is better)',
                 fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig1_master_heatmap.pdf', bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'fig1_master_heatmap.png', bbox_inches='tight')
    plt.close(); print("Fig 1 saved")

# ── FIG 2: Real data forecast plots (4 categories) ───────────────────────────
def fig_real_forecasts():
    cats = ['food','electronics','clothing','furniture']
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    for ax, cat in zip(axes.flatten(), cats):
        df = pd.read_csv(DATA_DIR / f'real_{cat}.csv', parse_dates=['ds'])
        series = df['y'].values.astype(float)
        split_i = int(len(series) * 0.8)
        train, test = series[:split_i], series[split_i:]
        test_dates = df['ds'].values[split_i:]

        ax.plot(df['ds'].values[:split_i], train,
                color='#CFD8DC', lw=1, label='Train')
        ax.plot(test_dates, test, color='black', lw=2, label='Actual', zorder=5)

        # ARIMA
        preds, _, _, _ = run_arima(train, test)
        ax.plot(test_dates, preds[:len(test)], '--',
                color=COLORS['ARIMA'], lw=1.8, label='ARIMA')

        # LSTM
        try:
            preds_l, _, _ = run_lstm(train, test)
            ax.plot(test_dates, preds_l[:len(test)], '-.',
                    color=COLORS['LSTM'], lw=1.5, label='LSTM')
        except: pass

        # Hybrid
        preds_h, _, _ = run_hybrid(train, test)
        ax.plot(test_dates, preds_h[:len(test)], '-',
                color=COLORS['Hybrid'], lw=1.8, label='Hybrid')

        ax.axvline(df['ds'].values[split_i], color='#607D8B',
                   lw=1.5, ls=':', alpha=0.8)
        ax.set_title(f'Real D-Mart — {cat.capitalize()} Category',
                     fontweight='bold')
        ax.set_ylabel('Daily Sales (aggregated)')
        ax.legend(loc='upper left', ncol=3, fontsize=8)

    axes[1][0].set_xlabel('Date'); axes[1][1].set_xlabel('Date')
    plt.suptitle('Real D-Mart Data: Forecast Comparison by Category\n'
                 '(Low-SNR regime — ARIMA competitive with complex models)',
                 fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig2_real_forecasts.pdf', bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'fig2_real_forecasts.png', bbox_inches='tight')
    plt.close(); print("Fig 2 saved")

# ── FIG 3: Regime characterization scatter ───────────────────────────────────
def fig_regime_scatter(results):
    """
    Key figure: CV (signal strength) vs Hybrid/ARIMA RMSE ratio.
    Shows the phase transition where complex models become useful.
    """
    from statsmodels.tsa.stattools import acf as acf_fn

    regimes = {
        'Real Food\n(D-Mart)':        ('Real_Food',      'real_food.csv',      '#E53935'),
        'Real Electronics\n(D-Mart)': ('Real_Electronics','real_electronics.csv','#E91E63'),
        'Real Clothing\n(D-Mart)':    ('Real_Clothing',   'real_clothing.csv',  '#9C27B0'),
        'Real Furniture\n(D-Mart)':   ('Real_Furniture',  'real_furniture.csv', '#673AB7'),
        'Simulated\nPrimary':         ('primary',         'primary.csv',        '#2196F3'),
        'Walmart\n(Weekly)':          ('walmart',         'walmart.csv',        '#4CAF50'),
        'M5\n(Intermittent)':         ('m5',              'm5.csv',             '#FF9800'),
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    cvs, ratios_hybrid, ratios_lstm = [], [], []
    labels_plot, colors_plot = [], []

    for label, (res_key, csv_file, color) in regimes.items():
        csv_path = DATA_DIR / csv_file
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if 'walmart' in csv_file:
            df = df[df['store']==1][['ds','y']] if 'store' in df.columns else df[['ds','y']]
        elif 'y' not in df.columns and 'Sales' in df.columns:
            df = df.rename(columns={'Sales':'y'})
        cv = df['y'].std() / df['y'].mean()

        m = results.get(res_key, {}).get('metrics', {})
        arima_r = m.get('ARIMA', {}).get('rmse', np.nan)
        hybrid_r = m.get('Hybrid', {}).get('rmse', np.nan)
        lstm_r   = m.get('LSTM',   {}).get('rmse', np.nan)

        if not np.isnan(arima_r) and arima_r > 0:
            cvs.append(cv)
            ratios_hybrid.append(hybrid_r / arima_r)
            ratios_lstm.append(lstm_r / arima_r)
            labels_plot.append(label)
            colors_plot.append(color)

    ax1 = axes[0]
    for cv, r, lbl, c in zip(cvs, ratios_hybrid, labels_plot, colors_plot):
        ax1.scatter(cv, r, color=c, s=120, zorder=5, edgecolors='white', lw=1.5)
        ax1.annotate(lbl, (cv, r), textcoords='offset points',
                     xytext=(5, 4), fontsize=8)
    ax1.axhline(1.0, color='#2196F3', lw=2, ls='--', label='ARIMA baseline')
    ax1.set_xlabel('Coefficient of Variation (CV = σ/μ)')
    ax1.set_ylabel('Hybrid RMSE / ARIMA RMSE')
    ax1.set_title('When Does Hybrid Beat ARIMA?\n(Ratio < 1 = Hybrid better)',
                  fontweight='bold')
    ax1.legend(fontsize=9)

    ax2 = axes[1]
    for cv, r, lbl, c in zip(cvs, ratios_lstm, labels_plot, colors_plot):
        ax2.scatter(cv, r, color=c, s=120, zorder=5, edgecolors='white', lw=1.5)
        ax2.annotate(lbl, (cv, r), textcoords='offset points',
                     xytext=(5, 4), fontsize=8)
    ax2.axhline(1.0, color='#F44336', lw=2, ls='--', label='ARIMA baseline')
    ax2.set_xlabel('Coefficient of Variation (CV = σ/μ)')
    ax2.set_ylabel('LSTM RMSE / ARIMA RMSE')
    ax2.set_title('When Does LSTM Beat ARIMA?\n(Ratio < 1 = LSTM better)',
                  fontweight='bold')
    ax2.legend(fontsize=9)

    plt.suptitle('Signal Strength (CV) as Predictor of Model Selection\n'
                 'Low CV → ARIMA sufficient | High CV → Complex models beneficial',
                 fontweight='bold', fontsize=12)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig3_regime_scatter.pdf', bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'fig3_regime_scatter.png', bbox_inches='tight')
    plt.close(); print("Fig 3 saved")

# ── FIG 4: Real data residual analysis ───────────────────────────────────────
def fig_real_residuals(results):
    cats = ['Real_Food','Real_Electronics','Real_Clothing','Real_Furniture']
    labels = ['Food','Electronics','Clothing','Furniture']
    models = ['ARIMA','SARIMA','XGBoost','LSTM','Hybrid']

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    for ax, cat, lbl in zip(axes, cats, labels):
        m_data = results.get(cat, {}).get('metrics', {})
        stds = [m_data.get(m, {}).get('residual_std', 0) for m in models]
        colors = [COLORS.get(m, '#607D8B') for m in models]
        bars = ax.bar(models, stds, color=colors, edgecolor='white')
        ax.set_title(f'{lbl}', fontweight='bold')
        ax.set_ylabel('Residual Std Dev')
        ax.set_xticklabels(models, rotation=35, ha='right')
        for bar, v in zip(bars, stds):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{v:.0f}', ha='center', fontsize=8)

    plt.suptitle('Residual Stability — Real D-Mart Data\n(Lower = more stable predictions)',
                 fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig4_real_residuals.pdf', bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'fig4_real_residuals.png', bbox_inches='tight')
    plt.close(); print("Fig 4 saved")

# ── FIG 5: Failure modes across all datasets ─────────────────────────────────
def fig_failure_all(results):
    datasets = ['Real_Food','Real_Electronics','Real_Clothing','Real_Furniture',
                'walmart','m5']
    labels   = ['Food\n(Real)','Electronics\n(Real)','Clothing\n(Real)','Furniture\n(Real)',
                'Walmart\n(Weekly)','M5\n(Interm.)']
    models   = ['SARIMA','Prophet','XGBoost','LSTM','Hybrid']

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for ax, ds_key, lbl in zip(axes.flatten(), datasets, labels):
        m_data = results.get(ds_key, {}).get('metrics', {})
        arima_r = m_data.get('ARIMA', {}).get('rmse', 1)
        rel = [m_data.get(m, {}).get('rmse', np.nan) / arima_r for m in models]
        colors = ['#4CAF50' if v <= 1 else '#F44336' for v in rel]
        bars = ax.bar(models, rel, color=colors, edgecolor='white')
        ax.axhline(1.0, color='#2196F3', lw=2, ls='--', label='ARIMA')
        ax.set_title(lbl, fontweight='bold')
        ax.set_ylabel('Relative RMSE vs ARIMA')
        ax.set_xticklabels(models, rotation=30, ha='right')
        for bar, v in zip(bars, rel):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.01, f'{v:.2f}x',
                        ha='center', fontsize=8)

    axes[0][0].legend(fontsize=9)
    plt.suptitle('When Does ARIMA Fail? — All Datasets\n'
                 'Green: model beats ARIMA | Red: model worse than ARIMA',
                 fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig5_failure_all.pdf', bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'fig5_failure_all.png', bbox_inches='tight')
    plt.close(); print("Fig 5 saved")

# ── FIG 6: DM test summary ────────────────────────────────────────────────────
def fig_dm_summary(results):
    datasets = ['Real_Food','Real_Clothing','walmart','m5']
    labels   = ['Real Food (D-Mart)','Real Clothing (D-Mart)',
                'Walmart (Weekly)','M5 (Intermittent)']
    models   = ['SARIMA','Prophet','XGBoost','LSTM','Hybrid']

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, ds_key, lbl in zip(axes.flatten(), datasets, labels):
        dm = results.get(ds_key, {}).get('dm_tests', {})
        pvals = [dm.get(m, {}).get('p_value', 1.0) for m in models]
        colors = ['#4CAF50' if p < 0.05 else '#FF9800' for p in pvals]
        ax.bar(models, pvals, color=colors, edgecolor='white')
        ax.axhline(0.05, color='#F44336', lw=2, ls='--', label='α=0.05')
        ax.set_title(lbl, fontweight='bold')
        ax.set_ylabel('DM Test p-value')
        ax.set_xticklabels(models, rotation=30, ha='right')
        ax.legend(fontsize=8)

    plt.suptitle('Diebold-Mariano Statistical Significance Tests\nvs ARIMA Baseline',
                 fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig6_dm_summary.pdf', bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'fig6_dm_summary.png', bbox_inches='tight')
    plt.close(); print("Fig 6 saved")

if __name__ == '__main__':
    FIGURES_DIR.mkdir(exist_ok=True)
    results = load_all_results()
    fig_master_heatmap(results)
    fig_real_forecasts()
    fig_regime_scatter(results)
    fig_real_residuals(results)
    fig_failure_all(results)
    fig_dm_summary(results)
    print("\nAll figures saved.")
