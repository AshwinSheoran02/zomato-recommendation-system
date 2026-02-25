"""
📊 Step 02 — Evaluate Model
============================
  1. Load trained model from models/lightgbm_model.pkl
  2. Load feature list from models/feature_list.json
  3. Load training_rows.csv & perform temporal split
  4. Baseline evaluation  (popularity ranking)
  5. Model evaluation     (predicted-probability ranking)
  6. Business metrics comparison
  7. Save charts → assets/figures/
     • precision_at_10.png
     • acceptance_rate.png
     • aov_comparison.png

Usage:
    python scripts/02_evaluate_model.py
"""

import json
import numpy as np
import pandas as pd
import pathlib
import warnings
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

ROOT = pathlib.Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
FIGS = ROOT / "assets" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

K = 10


# ══════════════════════════════════════════════════════════════════════
#  SHARED HELPERS  (imported logic identical to 01_train_model)
# ══════════════════════════════════════════════════════════════════════

def load_data():
    """Load training_rows.csv."""
    df = pd.read_csv(PROC / "training_rows.csv",
                     parse_dates=["order_timestamp"])
    print(f"✅ Loaded training_rows.csv  →  {df.shape}")
    return df


def temporal_split(df, ratio=0.80):
    """Temporal split keeping snapshots intact."""
    snap_ts = (df.groupby("snapshot_id")["order_timestamp"]
                 .first()
                 .sort_values())

    cutoff_idx  = int(len(snap_ts) * ratio)
    cutoff_date = snap_ts.iloc[cutoff_idx]

    train_snaps = set(snap_ts[snap_ts < cutoff_date].index)
    test_snaps  = set(snap_ts[snap_ts >= cutoff_date].index)

    train_df = df[df["snapshot_id"].isin(train_snaps)].copy()
    test_df  = df[df["snapshot_id"].isin(test_snaps)].copy()

    print(f"\n📅 Temporal split cutoff : {cutoff_date}")
    print(f"   Train : {len(train_df):>10,} rows  ({len(train_snaps):,} snapshots)")
    print(f"   Test  : {len(test_df):>10,} rows  ({len(test_snaps):,} snapshots)")
    return train_df, test_df


def prepare_features_for_eval(df, feature_cols):
    """
    Label-encode candidate_category and extract feature matrix.
    Uses the saved feature_cols to guarantee column alignment.
    """
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df["candidate_category"] = le.fit_transform(df["candidate_category"])
    X = df[feature_cols]
    return X


# ══════════════════════════════════════════════════════════════════════
#  EVALUATION HELPERS
# ══════════════════════════════════════════════════════════════════════

def _rank_topk(snap_id, grp, score_col, k):
    """Rank by *score_col* descending, return per-snapshot metrics."""
    ranked = grp.sort_values(score_col, ascending=False)
    top_k  = ranked.head(k)

    n_relevant = grp["label"].sum()
    hits       = top_k["label"].sum()

    return {
        "snapshot_id":     snap_id,
        "precision_at_k":  hits / k,
        "recall_at_k":     (hits / n_relevant) if n_relevant > 0 else 0.0,
        "any_hit":         int(hits > 0),
        "hit_revenue":     top_k.loc[top_k["label"] == 1, "candidate_price"].sum(),
        "n_hits":          hits,
        "cart_value":      grp["cart_value"].iloc[0],
        "cart_item_count": grp["cart_item_count"].iloc[0],
    }


def evaluate_baseline(test_df, k=K):
    """Rank candidates by candidate_popularity, take top-K per snapshot."""
    records = []
    for snap_id, grp in test_df.groupby("snapshot_id"):
        rec = _rank_topk(snap_id, grp, "candidate_popularity", k)
        records.append(rec)

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


def evaluate_model(model, test_df, X_test, k=K):
    """Rank candidates by model P(label=1), take top-K per snapshot."""
    test_df = test_df.copy()
    test_df["pred_prob"] = model.predict_proba(X_test)[:, 1]

    records = []
    for snap_id, grp in test_df.groupby("snapshot_id"):
        rec = _rank_topk(snap_id, grp, "pred_prob", k)
        records.append(rec)

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
#  BUSINESS METRICS
# ══════════════════════════════════════════════════════════════════════

def business_metrics(bl_res, ml_res):
    """
    Compare baseline vs model:
      • Precision@10 / Recall@10
      • Acceptance rate
      • Avg Order Value   (cart_value + reco revenue)
      • Avg items / order (cart_item_count + reco hits)
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
#  PLOTS
# ══════════════════════════════════════════════════════════════════════

def plot_comparisons(comp):
    """Save three bar charts comparing baseline vs model."""
    COLORS = ["#e74c3c", "#2ecc71"]

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
    _bar("Precision@10",        "precision_at_10.png",  fmt=".4f")
    _bar("Acceptance Rate",     "acceptance_rate.png",   fmt=".2%")
    _bar("Avg Order Value (₹)", "aov_comparison.png",   fmt=".1f", ylabel="₹")
    print("   ✅ All charts saved.\n")


# ══════════════════════════════════════════════════════════════════════
#  🏁  MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    # 1. Load model & feature list
    model_path = MODELS / "lightgbm_model.pkl"
    feat_path  = MODELS / "feature_list.json"

    print(f"📦 Loading model from {model_path}")
    model = joblib.load(model_path)

    with open(feat_path) as f:
        feature_cols = json.load(f)
    print(f"📦 Loaded feature list ({len(feature_cols)} features)")

    # 2. Load data & temporal split
    df = load_data()
    train_df, test_df = temporal_split(df)

    # 3. Prepare test features
    X_test = prepare_features_for_eval(test_df, feature_cols)

    # 4. Baseline evaluation
    bl_res, bl_prec, bl_rec, bl_acc = evaluate_baseline(test_df)

    # 5. Model evaluation
    ml_res, ml_prec, ml_rec, ml_acc = evaluate_model(model, test_df, X_test)

    # 6. Business metrics comparison
    comp = business_metrics(bl_res, ml_res)

    # 7. Save plots
    plot_comparisons(comp)

    print("🏁 Evaluation complete!")


if __name__ == "__main__":
    main()
