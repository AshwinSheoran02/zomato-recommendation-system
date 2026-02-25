# Zomato Cross-Sell Add-On Recommendation System

> **Intelligent real-time add-on recommendations that increase Average Order Value and acceptance rate at checkout.**

---

## 1. Problem Statement

Zomato's checkout flow leaves revenue on the table — customers see generic "you might also like" suggestions that rarely convert.  
**Goal:** Increase add-on revenue per order through intelligent, real-time, personalized recommendations.

---

## 2. How to Run

```bash
pip install numpy pandas faker tqdm scikit-learn lightgbm matplotlib joblib
```

> **Already have `models/lightgbm_model.pkl`?** Skip to Step 2 — data generation takes ~10 min and **is not needed** if the model file is present.

### Full Pipeline

| Step | Command | What it does | Time |
|:----:|---------|:-------------|:----:|
| **0** | `python scripts/00_make_data.py` | Generate all synthetic data | ~8 min |
| **1** | `python scripts/01_train_model.py` | Train LightGBM & save model | ~1 min |
| **2** | `python scripts/02_evaluate_model.py` | Evaluate model & produce charts | ~30 s |
| **3** | `python scripts/03_strategic_analysis.py` | Segments, cold-start, latency | ~30 s |

### Quick Run (model already exists)

If **`models/lightgbm_model.pkl`** and **`data/processed/training_rows.csv`** are present, run only the evaluation & analysis:

```bash
python scripts/02_evaluate_model.py          # → charts in assets/figures/
python scripts/03_strategic_analysis.py      # → strategic analysis charts
```

> **Steps 2 & 3 are independent** — run either or both.

---

## 3. Why Current Systems Fail

| Gap | Impact |
|-----|--------|
| **Popularity-based only** | Same suggestions for every user, regardless of context |
| **No personalization** | Ignores cart contents, time-of-day, cuisine preference |
| **Weak cold-start handling** | New users with < 5 orders get irrelevant picks |
| **No sequential understanding** | Doesn't adapt as items are added to cart |

---

## 4. Our Solution — 5 Strategic Layers

```
┌─────────────────────────────────────────────────────────┐
│  Layer 1 │  Predictive Ranking Model (LightGBM)        │
├──────────┼──────────────────────────────────────────────┤
│  Layer 2 │  Segment-Aware Optimization                  │
├──────────┼──────────────────────────────────────────────┤
│  Layer 3 │  Cold-Start Strategy                         │
├──────────┼──────────────────────────────────────────────┤
│  Layer 4 │  Sequential Cart Simulation                  │
├──────────┼──────────────────────────────────────────────┤
│  Layer 5 │  Real-Time Deployment Architecture           │
└──────────┴──────────────────────────────────────────────┘
```

- **Predictive Ranking** — LightGBM trained on 845K cart-item pairs with 10 contextual features, replacing static popularity with probability-scored ranking.
- **Segment-Aware** — Separate evaluation for high-value vs. casual customers; adaptive thresholds per segment.
- **Cold-Start** — Targeted strategy for users with fewer than 5 orders; lifts acceptance from 42% → 61%.
- **Sequential Cart Sim** — Re-ranks candidates after each add, modeling how the cart evolves in real-time.
- **Real-Time Architecture** — Sub-millisecond inference (p95 ≈ 1.08 ms) enabling live scoring at checkout.

---

## 5. Key Results

| Metric | Baseline | Ours | Lift |
|:-------|:--------:|:----:|:----:|
| **Acceptance Rate** | 52.9 % | 71.7 % | **+35 %** |
| **Avg Order Value** | ₹ 804 | ₹ 830 | **+3.3 %** |
| **Cold-Start Lift** | 42 % | 61 % | **+46 %** |
| **Latency (p95)** | — | 1.08 ms | **Real-time** |

---

## 6. Architecture

```
┌──────────┐     ┌────────────────┐     ┌───────────────┐     ┌──────────────┐     ┌─────────┐
│          │     │                │     │               │     │              │     │         │
│  User    ├────►│  Feature       ├────►│  LightGBM     ├────►│  Ranked      ├────►│ Display │
│  Cart    │     │  Engine        │     │  Model        │     │  Add-ons     │     │         │
│          │     │                │     │               │     │              │     │         │
└──────────┘     └────────────────┘     └───────────────┘     └──────────────┘     └────┬────┘
                                                                                        │
                        ┌───────────────────────────────────────────────────────────────┘
                        │
                        ▼
                 ┌──────────────┐
                 │  Feedback    │
                 │  Loop        │
                 └──────────────┘
```

**User Cart** → extract 10 real-time features → **LightGBM** scores every candidate → **Top-K ranked** add-ons displayed → user accept / reject feeds back into retraining.

---

## 7. Repository Structure

```
zomato-recommendation-system/
│
├── scripts/                              # All runnable code
│   ├── 00_make_data.py                   # One-click data generation pipeline
│   ├── 01_train_model.py                 # Train LightGBM & save artifacts
│   ├── 02_evaluate_model.py              # Evaluate & produce comparison charts
│   ├── 03_strategic_analysis.py          # Segments, cold-start, cart sim, latency
│   └── data_generation/                  # Sub-scripts called by 00_make_data
│       ├── 01_generate_base_tables.py
│       ├── 02_generate_orders.py
│       └── 03_build_training_table.py
│
├── data/
│   ├── raw/                              # users, restaurants, items, orders CSVs
│   └── processed/                        # training_rows, baseline_top10
│
├── models/                               # lightgbm_model.pkl, feature_list.json
├── assets/figures/                       # All output charts
├── docs/                                 # Documentation
├── LICENSE
└── README.md
```

---

## License

See [LICENSE](LICENSE) for details.
