"""
🧪 Step 05 — Model Training & Evaluation
=========================================
  1. Load training_rows.csv
  2. Temporal train/test split (80/20 by date)
  3. Feature preparation
  4. Baseline model (popularity ranking)
  5. LightGBM classifier training
  6. Ranking evaluation (Precision@10, Recall@10)
  7. Business metrics comparison
  8. Visualisation (bar charts → assets/figures/)

Usage:
    python scripts/05_model_train_eval.py
"""

import numpy as np
import pandas as pd
import pathlib
import warnings
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

ROOT = pathlib.Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
FIGS = ROOT / "assets" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

K = 10                                       # top-K for all evaluations


# ══════════════════════════════════════════════════════════════════════
#  1️⃣  LOAD DATA
# ══════════════════════════════════════════════════════════════════════
def load_data():
    """Load training_rows.csv, print shape & label distribution."""
    df = pd.read_csv(PROC / "training_rows.csv",
                     parse_dates=["order_timestamp"])

    print(f"✅ Loaded training_rows.csv  →  {df.shape}")
    print(f"\n📊 Label distribution:")
    print(df["label"].value_counts().to_string())
    print(f"\n   Positive rate: {df['label'].mean():.2%}")
    return df


# ══════════════════════════════════════════════════════════════════════
#  2️⃣  TEMPORAL TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════════════════════
def temporal_split(df):
    """
    Split by date so that test = last 20 % of calendar days.
    All rows from a given snapshot stay together → no data leakage.
    """
    # One timestamp per snapshot
    snap_ts = (df.groupby("snapshot_id")["order_timestamp"]
                 .first()
                 .sort_values())

    cutoff_idx  = int(len(snap_ts) * 0.80)
    cutoff_date = snap_ts.iloc[cutoff_idx]

    train_snaps = set(snap_ts[snap_ts < cutoff_date].index)
    test_snaps  = set(snap_ts[snap_ts >= cutoff_date].index)

    train_df = df[df["snapshot_id"].isin(train_snaps)].copy()
    test_df  = df[df["snapshot_id"].isin(test_snaps)].copy()

    print(f"\n📅 Temporal split cutoff : {cutoff_date}")
    print(f"   Train : {len(train_df):>10,} rows  ({len(train_snaps):,} snapshots)")
    print(f"   Test  : {len(test_df):>10,} rows  ({len(test_snaps):,} snapshots)")
    return train_df, test_df


# ══════════════════════════════════════════════════════════════════════
#  3️⃣  FEATURE PREPARATION
# ══════════════════════════════════════════════════════════════════════
def prepare_features(train_df, test_df):
    """
    • Label-encode candidate_category
    • Drop ID / timestamp / grouping columns
    • Return X_train, y_train, X_test, y_test
    """
    # ── Label-encode the one categorical feature ──────────────────────
    le = LabelEncoder()
    train_df["candidate_category"] = le.fit_transform(train_df["candidate_category"])
    test_df["candidate_category"]  = le.transform(test_df["candidate_category"])

    # ── Columns to drop (IDs, raw timestamps, grouping keys, target) ─
    drop_cols = [
        "snapshot_id", "order_id", "order_timestamp",
        "user_id", "restaurant_id", "candidate_item_id",
        "label",
    ]
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    X_train = train_df[feature_cols]
    y_train = train_df["label"]
    X_test  = test_df[feature_cols]
    y_test  = test_df["label"]

    print(f"\n🔢 Features ({len(feature_cols)}): {feature_cols}")
    print(f"   X_train : {X_train.shape}")
    print(f"   X_test  : {X_test.shape}")
    return X_train, y_train, X_test, y_test, feature_cols


# ══════════════════════════════════════════════════════════════════════
#  4️⃣  BASELINE MODEL  (rank by popularity → top K)
# ══════════════════════════════════════════════════════════════════════
def evaluate_baseline(test_df, k=K):
    """For each snapshot, rank candidates by candidate_popularity, take top-K."""
    records = []
    for snap_id, grp in test_df.groupby("snapshot_id"):
        ranked = grp.sort_values("candidate_popularity", ascending=False)
        top_k  = ranked.head(k)

        n_relevant = grp["label"].sum()
        hits       = top_k["label"].sum()

        records.append({
            "snapshot_id":     snap_id,
            "precision_at_k":  hits / k,
            "recall_at_k":     (hits / n_relevant) if n_relevant > 0 else 0.0,
            "any_hit":         int(hits > 0),
            "hit_revenue":     top_k.loc[top_k["label"] == 1, "candidate_price"].sum(),
            "n_hits":          hits,
            "cart_value":      grp["cart_value"].iloc[0],
            "cart_item_count": grp["cart_item_count"].iloc[0],
        })

    res  = pd.DataFrame(records)
    prec = res["precision_at_k"].mean()
    rec  = res["recall_at_k"].mean()
    acc  = res["any_hit"].mean()

    print(f"\n{'─' * 55}")
    print(f"  📊  BASELINE  (Popularity Ranking)")
    print(f"{'─' * 55}")
    print(f"   Precision@{k}    : {prec:.4f}")
    print(f"   Recall@{k}       : {rec:.4f}")
    print(f"   Acceptance Rate  : {acc:.2%}")
    return res, prec, rec, acc


# ══════════════════════════════════════════════════════════════════════
#  5️⃣  TRAIN LIGHTGBM
# ══════════════════════════════════════════════════════════════════════
def train_lgbm(X_train, y_train):
    """Simple LightGBM classifier — 100 trees, max_depth 6."""
    model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=SEED,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    print(f"\n🤖 LightGBM trained  (100 estimators, max_depth=6)")
    return model


# ══════════════════════════════════════════════════════════════════════
#  6️⃣  RANKING EVALUATION  (model predictions)
# ══════════════════════════════════════════════════════════════════════
def evaluate_model(model, test_df, X_test, k=K):
    """Rank candidates by predicted P(label=1), take top-K per snapshot."""
    test_df = test_df.copy()
    test_df["pred_prob"] = model.predict_proba(X_test)[:, 1]

    records = []
    for snap_id, grp in test_df.groupby("snapshot_id"):
        ranked = grp.sort_values("pred_prob", ascending=False)
        top_k  = ranked.head(k)

        n_relevant = grp["label"].sum()
        hits       = top_k["label"].sum()

        records.append({
            "snapshot_id":     snap_id,
            "precision_at_k":  hits / k,
            "recall_at_k":     (hits / n_relevant) if n_relevant > 0 else 0.0,
            "any_hit":         int(hits > 0),
            "hit_revenue":     top_k.loc[top_k["label"] == 1, "candidate_price"].sum(),
            "n_hits":          hits,
            "cart_value":      grp["cart_value"].iloc[0],
            "cart_item_count": grp["cart_item_count"].iloc[0],
        })

    res  = pd.DataFrame(records)
    prec = res["precision_at_k"].mean()
    rec  = res["recall_at_k"].mean()
    acc  = res["any_hit"].mean()

    print(f"\n{'─' * 55}")
    print(f"  📊  LIGHTGBM MODEL  (Predicted Probability Ranking)")
    print(f"{'─' * 55}")
    print(f"   Precision@{k}    : {prec:.4f}")
    print(f"   Recall@{k}       : {rec:.4f}")
    print(f"   Acceptance Rate  : {acc:.2%}")
    return res, prec, rec, acc


# ══════════════════════════════════════════════════════════════════════
#  7️⃣  BUSINESS METRICS SIMULATION
# ══════════════════════════════════════════════════════════════════════
def business_metrics(bl_res, ml_res):
    """
    Compare baseline vs model:
      • Precision@10 / Recall@10
      • Add-on acceptance rate
      • Avg Order Value   (cart_value + accepted-reco revenue)
      • Avg items / order (cart_item_count + accepted-reco items)
    """
    def _agg(res):
        prec  = res["precision_at_k"].mean()
        rec   = res["recall_at_k"].mean()
        acc   = res["any_hit"].mean()
        aov   = (res["cart_value"] + res["hit_revenue"]).mean()
        items = (res["cart_item_count"] + res["n_hits"]).mean()
        return prec, rec, acc, aov, items

    bl_prec, bl_rec, bl_acc, bl_aov, bl_items = _agg(bl_res)
    ml_prec, ml_rec, ml_acc, ml_aov, ml_items = _agg(ml_res)

    rows = {
        "Precision@10":        [bl_prec,  ml_prec],
        "Recall@10":           [bl_rec,   ml_rec],
        "Acceptance Rate":     [bl_acc,   ml_acc],
        "Avg Order Value (₹)": [bl_aov,   ml_aov],
        "Avg Items / Order":   [bl_items, ml_items],
    }
    comp = pd.DataFrame(rows, index=["Baseline", "Model"]).T
    comp["Lift"]   = comp["Model"] - comp["Baseline"]
    comp["Lift %"] = (comp["Lift"] / comp["Baseline"].abs() * 100).round(2)

    print("\n" + "=" * 70)
    print("  📊  BUSINESS METRICS — BASELINE  vs  MODEL")
    print("=" * 70)
    print(comp.to_string(float_format=lambda x: f"{x:.4f}"))
    print("=" * 70)
    return comp


# ══════════════════════════════════════════════════════════════════════
#  8️⃣  PLOTS
# ══════════════════════════════════════════════════════════════════════
def plot_comparisons(comp):
    """Save three bar charts comparing baseline vs model."""
    COLORS = ["#e74c3c", "#2ecc71"]           # red = baseline, green = model

    def _bar(metric, filename, fmt=".4f", ylabel=None):
        vals = comp.loc[metric, ["Baseline", "Model"]]
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(["Baseline", "Model"], vals, color=COLORS, width=0.45)
        for b in bars:
            ht = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, ht,
                    f"{ht:{fmt}}", ha="center", va="bottom", fontsize=12)
        ax.set_title(metric, fontsize=14, fontweight="bold")
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(vals) * 1.30)
        fig.tight_layout()
        fig.savefig(FIGS / filename, dpi=150)
        plt.close(fig)
        print(f"   📈 Saved {filename}")

    print(f"\n🎨 Saving charts to assets/figures/")
    _bar("Precision@10",        "precision_at_10.png",   fmt=".4f")
    _bar("Avg Order Value (₹)", "aov_comparison.png",    fmt=".1f", ylabel="₹")
    _bar("Acceptance Rate",     "acceptance_rate.png",    fmt=".2%")
    print("   ✅ All charts saved.\n")


# ══════════════════════════════════════════════════════════════════════
#  🏁  MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    # ── 1. Load ───────────────────────────────────────────────────────
    df = load_data()

    # ── 2. Temporal split ─────────────────────────────────────────────
    train_df, test_df = temporal_split(df)

    # ── 3. Feature preparation ────────────────────────────────────────
    X_train, y_train, X_test, y_test, feature_cols = \
        prepare_features(train_df, test_df)

    # ── 4. Baseline (popularity ranking) ──────────────────────────────
    bl_res, bl_prec, bl_rec, bl_acc = evaluate_baseline(test_df)

    # ── 5. Train LightGBM ────────────────────────────────────────────
    model = train_lgbm(X_train, y_train)

    # ── 6. Model ranking evaluation ───────────────────────────────────
    ml_res, ml_prec, ml_rec, ml_acc = evaluate_model(model, test_df, X_test)

    # ── 7. Business metrics comparison ────────────────────────────────
    comp = business_metrics(bl_res, ml_res)

    # ── 8. Plots ──────────────────────────────────────────────────────
    plot_comparisons(comp)

    print("🏁 Model training & evaluation complete!")


if __name__ == "__main__":
    main()
