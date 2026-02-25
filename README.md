# Zomato Cross-Sell Add-On Recommendation System

A machine-learning pipeline that recommends additional items ("add-ons") to
Zomato customers during checkout, increasing Average Order Value (AOV) and
acceptance rate compared to a popularity-only baseline.

---

## Repository Structure

```
zomato-recommendation-system/
│
├── scripts/                          # All runnable code
│   ├── 00_make_data.py               # One-click data generation pipeline
│   ├── 01_train_model.py             # Train LightGBM & save model artifacts
│   ├── 02_evaluate_model.py          # Evaluate model, produce comparison charts
│   ├── 03_strategic_analysis.py      # Segment, cold-start, cart sim, latency
│   └── data_generation/              # Sub-scripts called by 00_make_data.py
│       ├── 01_generate_base_tables.py    # → users, restaurants, items
│       ├── 02_generate_orders.py         # → orders, order_items
│       └── 03_build_training_table.py    # → training_rows, baseline_top10
│
├── data/
│   ├── raw/                          # Generated CSVs (users, restaurants, items, orders, order_items)
│   └── processed/                    # training_rows.csv, baseline_top10.csv
│
├── models/                           # Saved model artifacts
│   ├── lightgbm_model.pkl            # Trained LightGBM classifier
│   └── feature_list.json             # Feature names used during training
│
├── assets/figures/                   # Output charts (precision, acceptance, AOV, etc.)
│
├── archive/                          # Old notebooks & scripts (not part of pipeline)
│
├── docs/                             # Documentation
├── LICENSE
└── README.md
```

---

## Quick Start

### Prerequisites

- **Python 3.10+**
- Required packages: `numpy`, `pandas`, `faker`, `tqdm`, `scikit-learn`,
  `lightgbm`, `matplotlib`, `joblib`

```bash
pip install numpy pandas faker tqdm scikit-learn lightgbm matplotlib joblib
```

### Step 0 — Generate Data

Generates all synthetic tables (users, restaurants, items, orders, training
rows) from scratch. Output lands in `data/raw/` and `data/processed/`.

```bash
python scripts/00_make_data.py
```

### Step 1 — Train Model

Loads `training_rows.csv`, performs an 80/20 temporal split, trains a LightGBM
classifier, and saves the model + feature list to `models/`.

```bash
python scripts/01_train_model.py
```

### Step 2 — Evaluate Model

Loads the saved model, runs baseline (popularity) and model (LightGBM)
evaluations on the held-out test set, computes business metrics, and saves
comparison charts to `assets/figures/`.

```bash
python scripts/02_evaluate_model.py
```

### Step 3 — Strategic Analysis

Runs segment-level analysis, cold-start evaluation, sequential cart
simulation, and latency benchmarking. Saves additional charts to
`assets/figures/`.

```bash
python scripts/03_strategic_analysis.py
```

---

## Pipeline Overview

| Step | Script | What It Does | Output |
|------|--------|-------------|--------|
| 0 | `00_make_data.py` | Orchestrates data generation | `data/raw/*.csv`, `data/processed/*.csv` |
| 1 | `01_train_model.py` | Temporal split → LightGBM training | `models/lightgbm_model.pkl`, `models/feature_list.json` |
| 2 | `02_evaluate_model.py` | Baseline vs model evaluation + charts | `assets/figures/precision_at_10.png`, `acceptance_rate.png`, `aov_comparison.png` |
| 3 | `03_strategic_analysis.py` | Segments, cold-start, cart sim, latency | `assets/figures/segment_acceptance.png`, `cold_start_comparison.png` |

---

## Features Used (10)

| Feature | Description |
|---------|-------------|
| `cart_value` | Total value of items already in cart |
| `cart_item_count` | Number of items in cart |
| `has_drink` | Whether cart contains a drink |
| `has_dessert` | Whether cart contains a dessert |
| `hour_of_day` | Hour when order was placed |
| `weekday` | Day of week (0 = Mon, 6 = Sun) |
| `candidate_price` | Price of the candidate add-on item |
| `candidate_category` | Encoded category of candidate item |
| `candidate_popularity` | Historical order frequency of candidate |
| `matches_user_veg_pref` | Whether candidate matches user's veg preference |

---

## License

See [LICENSE](LICENSE) for details.
