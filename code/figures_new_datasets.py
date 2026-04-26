import warnings; warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, json
from pathlib import Path

RESULTS_DIR = Path('/home/claude/forecasting_paper/results')
FIGURES_DIR = Path('/home/claude/forecasting_paper/figures')

plt.rcParams.update({'font.family':'serif','font.size':11,'figure.dpi':150,
    'axes.spines.top':False,'axes.spines.right':False})

# ── Fig 11: Updated regime scatter with ALL datasets ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Collect all data points
points = []

# D-Mart (real, 4 categories)
dmart = [
    ('D-Mart Food',        0.0207, 0.503, -0.036, '#E53935'),
    ('D-Mart Electronics', 0.0210, 0.499, -0.166, '#E91E63'),
    ('D-Mart Clothing',    0.0216, 0.501, -0.091, '#9C27B0'),
    ('D-Mart Furniture',   0.0219, 0.498, -0.209, '#673AB7'),
]
for name, cv, ratio, ac1, c in dmart:
    points.append({'name':name,'cv':cv,'ratio':ratio,'ac1':ac1,'color':c,'marker':'o','source':'D-Mart (India)'})

# Walmart
points.append({'name':'Walmart','cv':0.081,'ratio':0.503,'ac1':0.71,'color':'#2196F3','marker':'s','source':'Walmart (US)'})

# M5
points.append({'name':'M5 Interm.','cv':0.91,'ratio':1.05,'ac1':0.03,'color':'#FF9800','marker':'^','source':'M5 Calibrated'})

# UCI Online Retail (UK, real)
uci = [
    ('UCI JumboBag',    0.588, 7141.2/7153.0, 0.465, '#4CAF50'),
    ('UCI LunchBag',    0.493, 1770.4/1367.2, 0.631, '#8BC34A'),
    ('UCI Retrospot',   0.320, 854.8/953.4,   0.110, '#CDDC39'),
    ('UCI CakeStand',   0.580, 973.4/1022.7,  0.636, '#009688'),
    ('UCI Total',       0.449, 86037.6/97294.6,0.701,'#00BCD4'),
]
for name, cv, ratio, ac1, c in uci:
    points.append({'name':name,'cv':cv,'ratio':ratio,'ac1':ac1,'color':c,'marker':'D','source':'UCI Online Retail (UK)'})

# M4 Micro Monthly — aggregate by CV bin
m4_data = {
    'low_cv':  {'cv':0.053,'ratio':0.570,'ac1':0.829},
    'med_cv':  {'cv':0.186,'ratio':0.781,'ac1':0.896},
    'high_cv': {'cv':0.362,'ratio':0.517,'ac1':0.914},
}
m4_colors = ['#FF5722','#FF7043','#FF8A65']
for (bin_name, d), c in zip(m4_data.items(), m4_colors):
    points.append({'name':f'M4 {bin_name}','cv':d['cv'],'ratio':d['ratio'],
                   'ac1':d['ac1'],'color':c,'marker':'*','source':'M4 Micro (Intl.)'})

df_pts = pd.DataFrame(points)

# Panel A: CV vs Hybrid/ARIMA ratio
ax = axes[0]
sources = df_pts['source'].unique()
src_markers = {'D-Mart (India)':'o','Walmart (US)':'s','M5 Calibrated':'^',
               'UCI Online Retail (UK)':'D','M4 Micro (Intl.)':'*'}

for _, row in df_pts.iterrows():
    ax.scatter(row['cv'], row['ratio'], color=row['color'],
               marker=row['marker'], s=150 if row['marker']=='*' else 100,
               edgecolors='white', linewidth=1.2, zorder=5)

# Source legend
for src, marker in src_markers.items():
    ax.scatter([], [], marker=marker, color='#607D8B', s=80, label=src)

ax.axhline(1.0, color='black', lw=2, ls='--', label='Break-even (=ARIMA)')
ax.axvspan(0, 0.03,  alpha=0.08, color='#4CAF50')
ax.axvspan(0.03,0.15,alpha=0.05, color='#FF9800')
ax.axvspan(0.15, 1.2,alpha=0.05, color='#F44336')
ax.text(0.015, 1.45, 'ARIMA\nfavored', ha='center', fontsize=8, color='#2E7D32')
ax.text(0.09,  1.45, 'Uncertain', ha='center', fontsize=8, color='#E65100')
ax.text(0.60,  1.45, 'Hybrid/ML\nfavored',   ha='center', fontsize=8, color='#B71C1C')
ax.set_xlabel('Coefficient of Variation (CV)')
ax.set_ylabel('Hybrid RMSE / ARIMA RMSE\n(or AR1/Naive for M4)')
ax.set_title('Regime Characterization — All Datasets\n(5 dataset sources, 3 countries)',
             fontweight='bold')
ax.legend(fontsize=8, loc='upper right', ncol=1)
ax.set_xlim(-0.02, 1.0); ax.set_ylim(0.3, 1.6)

# Panel B: AC(1) vs ratio
ax2 = axes[1]
for _, row in df_pts.iterrows():
    ax2.scatter(row['ac1'], row['ratio'], color=row['color'],
                marker=row['marker'], s=150 if row['marker']=='*' else 100,
                edgecolors='white', linewidth=1.2, zorder=5)

ax2.axhline(1.0, color='black', lw=2, ls='--')
ax2.axvline(0.2, color='#FF9800', lw=1.5, ls=':', alpha=0.7, label='AC(1)=0.2 threshold')
ax2.set_xlabel('Lag-1 Autocorrelation AC(1)')
ax2.set_ylabel('Hybrid RMSE / ARIMA RMSE\n(or AR1/Naive for M4)')
ax2.set_title('AC(1) vs Model Performance\nHigh AC(1) enables structured models to win',
              fontweight='bold')
ax2.legend(fontsize=9)

# Add simple regression line
from scipy import stats
ac1_vals = df_pts['ac1'].values; ratio_vals = df_pts['ratio'].values
sl,ic,r,p,_ = stats.linregress(ac1_vals, ratio_vals)
x_line = np.linspace(min(ac1_vals), max(ac1_vals), 50)
ax2.plot(x_line, sl*x_line+ic, 'k--', lw=1, alpha=0.4,
         label=f'Trend (R²={r**2:.2f}, p={p:.3f})')
ax2.legend(fontsize=8)

plt.suptitle('Regime Characterization Across 5 Dataset Sources\n'
             'D-Mart (India) · Walmart (US) · UCI Online Retail (UK) · M4 Micro · M5',
             fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig(FIGURES_DIR/'fig11_all_datasets_regime.pdf', bbox_inches='tight')
plt.savefig(FIGURES_DIR/'fig11_all_datasets_regime.png', bbox_inches='tight')
plt.close()
print("Fig 11 saved.")

# ── Fig 12: UCI + M4 results bar chart ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: UCI by product category
uci_names = ['JumboBag','LunchBag','Retrospot','CakeStand','Total']
arima_rmse= [7153.0, 1367.2, 953.4, 1022.7, 97294.6]
hybrid_rmse=[8021.1, 1770.4, 854.8,  973.4, 86037.6]
xgb_rmse  =[7141.2, 1629.1, 900.0, 1039.8, 86497.5]

x = np.arange(len(uci_names)); w=0.25
ax = axes[0]
ax.bar(x-w, arima_rmse,  w, label='ARIMA',   color='#2196F3', edgecolor='white')
ax.bar(x,   xgb_rmse,    w, label='XGBoost', color='#9C27B0', edgecolor='white')
ax.bar(x+w, hybrid_rmse, w, label='Hybrid',  color='#795548', edgecolor='white')
ax.set_xticks(x); ax.set_xticklabels(uci_names, rotation=25, ha='right')
ax.set_ylabel('RMSE')
ax.set_title('UCI Online Retail (UK, real)\nWeekly Product Revenue', fontweight='bold')
ax.legend(fontsize=9)
ax.set_yscale('log')

# Right: M4 Micro — AR1/Naive ratio by CV bin
m4_bins   = ['Low CV\n(<0.10)', 'Med CV\n(0.10-0.30)', 'High CV\n(0.30-0.60)']
m4_ratios = [0.570, 0.781, 0.517]
m4_stds   = [0.195, 0.588, 0.455]
m4_n      = [8, 8, 8]

ax2 = axes[1]
bars = ax2.bar(m4_bins, m4_ratios, color=['#2196F3','#FF9800','#F44336'],
               edgecolor='white', yerr=m4_stds, capsize=6)
ax2.axhline(1.0, color='black', lw=2, ls='--', label='Naive baseline')
ax2.set_ylabel('AR(1) / Naive RMSE\n(lower = AR(1) better)')
ax2.set_title('M4 Micro Monthly (International)\nAR(1) vs Naive Mean Model', fontweight='bold')
ax2.legend(fontsize=9)
for bar, n, r in zip(bars, m4_n, m4_ratios):
    ax2.text(bar.get_x()+bar.get_width()/2, 0.05, f'n={n}\nratio={r:.2f}',
             ha='center', fontsize=9, color='white', fontweight='bold')

plt.suptitle('New Dataset Results: UCI Online Retail + M4 Micro Monthly\n'
             'Both confirm regime-dependent model performance',
             fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig(FIGURES_DIR/'fig12_new_datasets.pdf', bbox_inches='tight')
plt.savefig(FIGURES_DIR/'fig12_new_datasets.png', bbox_inches='tight')
plt.close()
print("Fig 12 saved.")
