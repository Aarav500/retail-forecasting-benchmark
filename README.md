# Regime-Dependent Performance of ARIMA and Modern Forecasting Methods
### An Empirical Benchmark on Small-Scale Retail Demand Data

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![TMLR Submission](https://img.shields.io/badge/TMLR-Under%20Review-orange)](https://jmlr.org/tmlr/)

**Aarav Shah** · University of California, Riverside · `ashah264@ucr.edu`  
GitHub: [github.com/Aarav500/retail-forecasting-benchmark](https://github.com/Aarav500/retail-forecasting-benchmark)

---

## Summary

We benchmark six forecasting methods across **5 dataset sources, 3 countries, 34 time series**:

| Dataset | Source | Country | n | Real? |
|:---|:---|:---|---:|:---|
| D-Mart (4 category series) | Single retail store | India | 181/series | ✓ Real |
| UCI Online Retail (5 series) | E-commerce retailer | UK | 53/series | ✓ Real |
| M4 Micro Monthly (24 series) | M4 competition | International | 68–197 | ✓ Real |
| Walmart (1 series) | Calibrated to Kaggle data | US | 143 | Calibrated |
| M5 Intermittent (1 series) | Calibrated to M5 stats | — | 365 | Calibrated |

**Central finding**: Model selection must be regime-aware.

| Regime | CV | AC(1) | Best model |
|:---|:---|:---|:---|
| Low-SNR (D-Mart daily) | ~0.02 | <0.21 | **ARIMA** |
| High-variance (UCI, Walmart) | 0.32–0.59 | >0.40 | **Hybrid (ARIMA+XGBoost)** |
| High-autocorrelation (M4 monthly) | any | >0.60 | **Structured model** |
| Intermittent (M5) | N/A | ~0 | Specialized methods |

---

## Key Results

### D-Mart (India) — ARIMA wins all 4 categories (CV≈0.021)

| Model | Food | Electronics | Clothing | Furniture |
|:---|---:|---:|---:|---:|
| **ARIMA** | **255.76** | **227.20** | 238.78 | **219.06** |
| LSTM | 262.82 | 230.06 | **235.72** | 219.88 |
| Hybrid | 312.37 | 240.65 | 274.02 | 262.81 |
| Prophet | 530.09 | 278.02 | 723.46 | 287.82 |

### UCI Online Retail (UK) — Hybrid wins 4/5 at CV 0.32–0.59

| Series | CV | ARIMA | Hybrid | Winner |
|:---|---:|---:|---:|:---|
| Jumbo Bag | 0.59 | 7,153 | 8,021 | XGBoost |
| Lunch Bag | 0.49 | 1,367 | 1,770 | **ARIMA** |
| Red Retrospot | 0.32 | 953 | **855** | Hybrid |
| Regency Cakestand | 0.58 | 1,023 | **973** | Hybrid |
| Store Total | 0.45 | 97,295 | **86,038** | Hybrid |

### M4 Micro Monthly — AR(1) beats naive in 21/24 series

Mean AR(1)/Naive ratio: **0.623** (37.7% RMSE reduction). Consistent across all CV bins when AC(1)≈0.88. Confirms AC(1) is the primary predictor of structured model advantage.

### Walmart weekly — Hybrid wins by 49.6%

ARIMA: 165.23 → Hybrid: **83.15**. All DM tests: p<0.001.

---

## Quickstart

```bash
git clone https://github.com/Aarav500/retail-forecasting-benchmark
cd retail-forecasting-benchmark
pip install -r requirements.txt

# Run D-Mart experiments (~45 min, CPU only)
python code/real_experiment.py

# Run UCI experiments
python code/figures_new_datasets.py

# Run M4 experiments
python code/m4_experiment.py

# Run ablation study
python code/ablation.py

# Generate all figures
python code/figures_v2.py
python code/figures_new_datasets.py
```

---

## Repository Structure

```
retail-forecasting-benchmark/
├── paper.pdf / paper.tex          # Full paper (TMLR submission)
├── requirements.txt
├── LICENSE                        # MIT
├── configs/
│   └── hyperparams.yaml           # All hyperparameters
├── code/
│   ├── experiment.py              # Main benchmark pipeline
│   ├── real_experiment.py         # D-Mart experiments
│   ├── figures_v2.py              # Main paper figures (Fig 1–10)
│   ├── figures_new_datasets.py    # UCI + regime figures (Fig 11–12)
│   ├── m4_experiment.py           # M4 Micro Monthly experiments
│   └── ablation.py                # Training window ablation
├── data/
│   ├── real_food/electronics/clothing/furniture.csv   # D-Mart (real)
│   ├── uci_*.csv                  # UCI Online Retail (real, UK)
│   ├── m4_sample_ids.csv          # M4 series IDs (24 sampled)
│   ├── m4_monthly_train.csv       # M4 training data
│   ├── walmart.csv                # Walmart-calibrated weekly
│   └── m5.csv                     # M5-calibrated intermittent
├── figures/                       # All 12 paper figures (PNG + PDF)
├── results/                       # JSON results (all experiments)
└── notebooks/
    └── exploration.ipynb          # EDA and result exploration
```

---

## Reproducibility

- Fixed random seeds: `numpy.random.seed(42)`, `tf.random.set_seed(42)`
- Strict temporal 80/20 split — no look-ahead
- Practitioner-default hyperparameters throughout (`configs/hyperparams.yaml`)
- CPU-only — no GPU required
- All DM tests: two-sided, Newey-West variance, squared-error loss, p<0.001

---

## Citation

```bibtex
@article{shah2025forecasting,
  title={Regime-Dependent Performance of {ARIMA} and Modern Forecasting Methods:
         An Empirical Benchmark on Small-Scale Retail Demand Data},
  author={Shah, Aarav},
  journal={Transactions on Machine Learning Research},
  year={2025},
  url={https://github.com/Aarav500/retail-forecasting-benchmark}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
