# When Does Classical Forecasting Fail?
### A Rigorous Multi-Regime Benchmark of ARIMA and Modern Methods on Small-Scale Retail Demand Data

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![TMLR Submission](https://img.shields.io/badge/TMLR-Under%20Review-orange)](https://jmlr.org/tmlr/)

**Aarav Shah** · University of California, Riverside · `ashah264@ucr.edu`

---

## TL;DR

On **real Indian retail data** (D-Mart, 800 SKUs, 4 categories, 181 days), classical ARIMA **beats or matches all** complex models — LSTM, XGBoost, Prophet, and our own Hybrid. On **high-variance structured data** (Walmart weekly), our Hybrid (ARIMA+XGBoost) reduces RMSE by **49.6%** over ARIMA. We introduce the **coefficient of variation (CV)** as a training-free model selection criterion.

| CV of demand | Recommended model |
|:---|:---|
| < 0.03 (e.g. new retail store, daily data) | **ARIMA** (mean model) |
| 0.03 – 0.10 | **SARIMA** |
| > 0.10 (e.g. Walmart weekly) | **Hybrid (ARIMA+XGBoost)** |
| Zero-fraction > 20% | Croston / INARMA (intermittent demand) |

---

## Key Results

### Real D-Mart Data — Low-SNR Regime (CV ≈ 0.021)

| Model | Food | Electronics | Clothing | Furniture |
|:---|---:|---:|---:|---:|
| **ARIMA** | **255.76** | **227.20** | 238.78 | **219.06** |
| SARIMA | **255.76** | 229.52 | 238.78 | **219.06** |
| Prophet | 530.09 | 278.02 | 723.46 | 287.82 |
| XGBoost | 304.08 | 249.26 | 271.07 | 251.55 |
| LSTM | 262.82 | 230.06 | **235.72** | 219.88 |
| Hybrid | 312.37 | 240.65 | 274.02 | 262.81 |

*ARIMA wins 15/16 category×model comparisons. All differences statistically significant under DM test (p < 0.001).*

### Walk-Forward CV — ARIMA wins every fold

| Category | ARIMA | XGBoost | Hybrid |
|:---|---:|---:|---:|
| Food | **204.4** | 230.6 | 235.4 |
| Electronics | **195.2** | 206.8 | 209.7 |
| Clothing | **246.9** | 259.4 | 266.1 |
| Furniture | **237.1** | 265.8 | 263.3 |

### SKU-Level — ARIMA wins 17/20 individual SKUs

ARIMA dominates even at SKU level (CV = 0.26–0.33), confirming the result is not an artifact of category aggregation.

### Walmart Weekly — Hybrid wins (high-variance regime)

| Model | RMSE |
|:---|---:|
| **Hybrid** | **83.15** ← 49.6% better than ARIMA |
| SARIMA | 123.13 |
| ARIMA | 165.23 |
| XGBoost | 292.34 |
| LSTM | 907.65 |
| Prophet | 944.93 |

---

## Models

| Model | Description | Reference |
|:---|:---|:---|
| ARIMA | Auto-selected via AIC (`auto_arima`) | Box & Jenkins (2015) |
| SARIMA | Seasonal ARIMA | Box & Jenkins (2015) |
| Prophet | Additive trend+seasonality | Taylor & Letham (2018) |
| XGBoost | Gradient boosted trees, 14-lag features | Chen & Guestrin (2016) |
| LSTM | 2-layer LSTM, 14-lag window | Hochreiter & Schmidhuber (1997) |
| **Hybrid** | ARIMA + XGBoost on residuals | **This paper** |

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/aaravshah/retail-forecasting-benchmark
cd retail-forecasting-benchmark

# 2. Install (Python 3.10+)
pip install -r requirements.txt

# 3. Run all model × dataset experiments (~45 min, CPU only)
python code/experiment.py

# 4. Run real D-Mart experiments
python code/real_experiment.py

# 5. Generate all 9 figures
python code/figures_v2.py

# 6. Run ablation study (training window size)
python code/ablation.py
python code/ablation.py --lstm   # include LSTM, adds ~80s

# 7. Explore results interactively
jupyter notebook notebooks/exploration.ipynb
```

---

## Repository Structure

```
retail-forecasting-benchmark/
├── README.md
├── requirements.txt
├── LICENSE                        # MIT
├── configs/
│   └── hyperparams.yaml           # All hyperparameters (fully specified)
├── code/
│   ├── experiment.py              # All models + all datasets
│   ├── real_experiment.py         # D-Mart category-level experiments
│   ├── figures_v2.py              # All 9 paper figures
│   └── ablation.py                # Training window size ablation
├── data/
│   ├── real_food.csv              # D-Mart Food (category agg, 181 days)
│   ├── real_electronics.csv       # D-Mart Electronics
│   ├── real_clothing.csv          # D-Mart Clothing
│   ├── real_furniture.csv         # D-Mart Furniture
│   ├── walmart.csv                # Walmart-style weekly data
│   └── m5.csv                     # M5-style intermittent demand
├── figures/                       # All 9 paper figures (PNG + PDF)
├── results/                       # JSON results (auto-generated)
│   ├── Real_Food_results.json
│   ├── Real_Electronics_results.json
│   ├── Real_Clothing_results.json
│   ├── Real_Furniture_results.json
│   ├── walmart_results.json
│   ├── m5_results.json
│   ├── walkforward_results.json
│   ├── sku_level_results.json
│   └── ablation_results.json
├── notebooks/
│   └── exploration.ipynb          # Interactive EDA and result exploration
├── paper.tex                      # Full LaTeX source
└── paper.pdf                      # Compiled paper
```

---

## Reproducibility

All results are fully reproducible:

- Fixed random seeds: `numpy.random.seed(42)`, `tf.random.set_seed(42)`
- Strict temporal 80/20 train/test split — no look-ahead
- All preprocessing (scaling, lag features) computed on training data only
- CPU-only — no GPU required (~45 min total runtime)
- All hyperparameters documented in `configs/hyperparams.yaml`
- DM test: two-sided, Newey-West variance, squared-error loss

---

## Citation

```bibtex
@article{shah2025forecasting,
  title={When Does Classical Forecasting Fail? A Rigorous Multi-Regime
         Benchmark of ARIMA and Modern Methods on Small-Scale Retail Demand Data},
  author={Shah, Aarav},
  journal={Transactions on Machine Learning Research},
  year={2025},
  url={https://github.com/aaravshah/retail-forecasting-benchmark}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
