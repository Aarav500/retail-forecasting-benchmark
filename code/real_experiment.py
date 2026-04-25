"""
Real data experiments using actual D-Mart retail dataset.
Runs all 6 models across 4 product categories + total.
"""
import warnings; warnings.filterwarnings('ignore')
import sys
sys.path.insert(0, '/home/claude/forecasting_paper/code')
from experiment import *

DATA_DIR    = Path("/home/claude/forecasting_paper/data")
RESULTS_DIR = Path("/home/claude/forecasting_paper/results")

def run_real_experiments():
    RESULTS_DIR.mkdir(exist_ok=True)

    categories = ['food', 'electronics', 'clothing', 'furniture']
    all_results = {}

    for cat in categories:
        df = pd.read_csv(DATA_DIR / f'real_{cat}.csv', parse_dates=['ds'])
        df = df[['ds', 'y']].copy()
        label = f'Real_{cat.capitalize()}'

        results, train, test, series = run_dataset_experiment(df, label, freq='D', split=0.8)
        all_results[label] = results

    # Summary
    print("\n" + "="*90)
    print("SUMMARY: Real D-Mart Data — RMSE across categories")
    print("="*90)
    models   = ['ARIMA','SARIMA','Prophet','XGBoost','LSTM','Hybrid']
    datasets = [f'Real_{c.capitalize()}' for c in categories]
    header   = f"{'Model':<20}" + "".join(f"{d:<22}" for d in datasets)
    print(header)
    print("-"*90)
    for m in models:
        row = f"{m:<20}"
        for d in datasets:
            v = all_results[d].get(m, {}).get('rmse', float('nan'))
            row += f"{v:<22.4f}"
        print(row)

    # Key finding: coefficient of variation to characterize regime
    print("\n=== Regime Characterization (CV = std/mean) ===")
    for cat in categories:
        df = pd.read_csv(DATA_DIR / f'real_{cat}.csv', parse_dates=['ds'])
        cv = df['y'].std() / df['y'].mean()
        from statsmodels.tsa.stattools import acf
        ac = acf(df['y'].values, nlags=7, fft=False)
        print(f"  {cat.capitalize()}: CV={cv:.3f}, AC(1)={ac[1]:.3f}, AC(7)={ac[7]:.3f}")

    return all_results

if __name__ == '__main__':
    results = run_real_experiments()
