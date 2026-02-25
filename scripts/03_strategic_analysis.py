"""
🔬 Step 03 — Strategic Analysis
=================================
  1. Segment analysis        (high-value vs casual users)
  2. Cold-start evaluation   (new users with few orders)
  3. Sequential cart simulation  (incremental add-ons)
  4. Latency benchmark       (per-snapshot inference time)

  Saves:
     • assets/figures/segment_acceptance.png
     • assets/figures/cold_start_comparison.png

Usage:
    python scripts/03_strategic_analysis.py
"""

import json
import time
import numpy as np
import pandas as pd
import pathlib
import warnings
import joblib
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

ROOT = pathlib.Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
RAW  = ROOT / "data" / "raw"
MODELS = ROOT / "models"
FIGS = ROOT / "assets" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

K = 10


# ══════════════════════════════════════════════════════════════════════
#  SHARED HELPERS
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
    """Label-encode candidate_category and extract feature matrix."""
    le = LabelEncoder()
    df["candidate_category"] = le.fit_transform(df["candidate_category"])
    X = df[feature_cols]
    return X


def rank_topk_metrics(grp, score_col, k=K):
    """Rank by *score_col* descending, return per-snapshot metrics dict."""
    ranked = grp.sort_values(score_col, ascending=False)
    top_k  = ranked.head(k)

    n_relevant = grp["label"].sum()
    hits       = top_k["label"].sum()

    return {
        "snapshot_id":     grp.name if hasattr(grp, "name") else grp["snapshot_id"].iloc[0],
        "precision_at_k":  hits / k,
        "recall_at_k":     (hits / n_relevant) if n_relevant > 0 else 0.0,
        "any_hit":         int(hits > 0),
        "hit_revenue":     top_k.loc[top_k["label"] == 1, "candidate_price"].sum(),
        "n_hits":          hits,
        "cart_value":      grp["cart_value"].iloc[0],
        "cart_item_count": grp["cart_item_count"].iloc[0],
    }


# ══════════════════════════════════════════════════════════════════════
#  1️⃣  SEGMENT ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def segment_analysis(model, test_df, feature_cols, k=K):
    """
    Split test users into HIGH-VALUE (top-25 % cart_value) vs CASUAL
    and compare model acceptance rate across segments.
    """
    print(f"\n{'═' * 60}")
    print("  🔍  SEGMENT ANALYSIS  (High-Value vs Casual)")
    print(f"{'═' * 60}")

    X_test = prepare_features_for_eval(test_df.copy(), feature_cols)
    test_df = test_df.copy()
    test_df["pred_prob"] = model.predict_proba(X_test)[:, 1]

    # Determine cart-value quantile per snapshot
    snap_cv = test_df.groupby("snapshot_id")["cart_value"].first()
    q75 = snap_cv.quantile(0.75)

    high_snaps   = set(snap_cv[snap_cv >= q75].index)
    casual_snaps = set(snap_cv[snap_cv <  q75].index)

    segments = {"High-Value": high_snaps, "Casual": casual_snaps}
    seg_results = {}

    for seg_name, snap_set in segments.items():
        seg_df = test_df[test_df["snapshot_id"].isin(snap_set)]
        records = []
        for snap_id, grp in seg_df.groupby("snapshot_id"):
            # Model ranking
            rec = rank_topk_metrics(grp, "pred_prob", k)
            records.append(rec)

        res = pd.DataFrame(records)
        prec = res["precision_at_k"].mean()
        acc  = res["any_hit"].mean()
        aov  = (res["cart_value"] + res["hit_revenue"]).mean()

        seg_results[seg_name] = {"prec": prec, "acc": acc, "aov": aov,
                                 "n_snapshots": len(snap_set)}

        print(f"\n   [{seg_name}]  ({len(snap_set):,} snapshots)")
        print(f"      Precision@{k}   : {prec:.4f}")
        print(f"      Acceptance Rate : {acc:.2%}")
        print(f"      Avg Order Value : ₹{aov:,.1f}")

    # ── Plot ──────────────────────────────────────────────────────────
    _plot_segment_acceptance(seg_results)
    return seg_results


def _plot_segment_acceptance(seg_results):
    labels = list(seg_results.keys())
    accs   = [seg_results[s]["acc"] for s in labels]
    colors = ["#3498db", "#95a5a6"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, accs, color=colors, width=0.45)
    for b in bars:
        ht = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, ht,
                f"{ht:.2%}", ha="center", va="bottom", fontsize=12)
    ax.set_title("Acceptance Rate by User Segment", fontsize=14, fontweight="bold")
    ax.set_ylabel("Acceptance Rate")
    ax.set_ylim(0, max(accs) * 1.30)
    fig.tight_layout()
    fig.savefig(FIGS / "segment_acceptance.png", dpi=150)
    plt.close(fig)
    print(f"\n   📈 Saved segment_acceptance.png")


# ══════════════════════════════════════════════════════════════════════
#  2️⃣  COLD-START EVALUATION
# ══════════════════════════════════════════════════════════════════════

def cold_start_evaluation(model, test_df, feature_cols, k=K):
    """
    Evaluate model on COLD-START users (≤ 5 past orders in training set)
    vs WARM users (> 5 past orders).

    Uses orders.csv to count historical orders per user.
    """
    print(f"\n{'═' * 60}")
    print("  🧊  COLD-START EVALUATION")
    print(f"{'═' * 60}")

    # Count orders per user from raw data
    orders = pd.read_csv(RAW / "orders.csv")
    user_order_count = orders.groupby("user_id").size().rename("n_orders")

    # Map user_id → order count in test set
    test_df = test_df.copy()
    test_df = test_df.merge(user_order_count, on="user_id", how="left")
    test_df["n_orders"] = test_df["n_orders"].fillna(0).astype(int)

    X_test = prepare_features_for_eval(test_df.copy(), feature_cols)
    test_df["pred_prob"] = model.predict_proba(X_test)[:, 1]

    cold_threshold = 5
    cold_snaps = set(test_df.loc[test_df["n_orders"] <= cold_threshold, "snapshot_id"])
    warm_snaps = set(test_df.loc[test_df["n_orders"] >  cold_threshold, "snapshot_id"])

    cohorts = {"Cold-Start": cold_snaps, "Warm": warm_snaps}
    cohort_results = {}

    for cohort_name, snap_set in cohorts.items():
        coh_df = test_df[test_df["snapshot_id"].isin(snap_set)]
        records = []
        for snap_id, grp in coh_df.groupby("snapshot_id"):
            # Model-ranked
            model_rec = rank_topk_metrics(grp, "pred_prob", k)
            model_rec["source"] = "Model"

            # Baseline (popularity)
            bl_rec = rank_topk_metrics(grp, "candidate_popularity", k)
            bl_rec["source"] = "Baseline"

            records.append(model_rec)
            records.append(bl_rec)

        res = pd.DataFrame(records)

        model_acc = res.loc[res["source"] == "Model",    "any_hit"].mean()
        bl_acc    = res.loc[res["source"] == "Baseline", "any_hit"].mean()

        cohort_results[cohort_name] = {
            "model_acc": model_acc,
            "baseline_acc": bl_acc,
            "n_snapshots": len(snap_set),
        }

        print(f"\n   [{cohort_name}]  ({len(snap_set):,} snapshots)")
        print(f"      Baseline  Acceptance : {bl_acc:.2%}")
        print(f"      Model     Acceptance : {model_acc:.2%}")

    # ── Plot ──────────────────────────────────────────────────────────
    _plot_cold_start(cohort_results)
    return cohort_results


def _plot_cold_start(cohort_results):
    cohorts  = list(cohort_results.keys())
    bl_accs  = [cohort_results[c]["baseline_acc"] for c in cohorts]
    ml_accs  = [cohort_results[c]["model_acc"]    for c in cohorts]

    x = np.arange(len(cohorts))
    width = 0.30

    fig, ax = plt.subplots(figsize=(7, 4))
    bars1 = ax.bar(x - width / 2, bl_accs, width, label="Baseline", color="#e74c3c")
    bars2 = ax.bar(x + width / 2, ml_accs, width, label="Model",    color="#2ecc71")

    for bars in [bars1, bars2]:
        for b in bars:
            ht = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, ht,
                    f"{ht:.2%}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(cohorts)
    ax.set_ylabel("Acceptance Rate")
    ax.set_title("Cold-Start vs Warm Users", fontsize=14, fontweight="bold")
    ax.legend()
    ax.set_ylim(0, max(bl_accs + ml_accs) * 1.35)
    fig.tight_layout()
    fig.savefig(FIGS / "cold_start_comparison.png", dpi=150)
    plt.close(fig)
    print(f"\n   📈 Saved cold_start_comparison.png")


# ══════════════════════════════════════════════════════════════════════
#  3️⃣  SEQUENTIAL CART SIMULATION
# ══════════════════════════════════════════════════════════════════════

def sequential_cart_simulation(model, test_df, feature_cols, k=K,
                                n_samples=200, max_rounds=3):
    """
    Simulate a multi-round add-on loop:
      Round 1 → recommend top-K → "accept" top-1 hit → update cart
      Round 2 → re-recommend on updated cart → repeat
      …
    Reports avg accepted items across rounds.
    """
    print(f"\n{'═' * 60}")
    print(f"  🔄  SEQUENTIAL CART SIMULATION  ({max_rounds} rounds, {n_samples} snapshots)")
    print(f"{'═' * 60}")

    snap_ids = test_df["snapshot_id"].unique()
    rng = np.random.default_rng(SEED)
    sample_snaps = rng.choice(snap_ids, size=min(n_samples, len(snap_ids)),
                              replace=False)

    round_accepts = {r: [] for r in range(1, max_rounds + 1)}

    for snap_id in sample_snaps:
        grp = test_df[test_df["snapshot_id"] == snap_id].copy()
        accepted = 0

        for rnd in range(1, max_rounds + 1):
            # Adjust cart features to reflect accepted items
            grp["cart_item_count"] = grp["cart_item_count"] + accepted

            X = prepare_features_for_eval(grp.copy(), feature_cols)
            probs = model.predict_proba(X)[:, 1]
            grp["pred_prob"] = probs

            top_k = grp.sort_values("pred_prob", ascending=False).head(k)
            hits  = top_k[top_k["label"] == 1]

            if len(hits) > 0:
                accepted += 1
                # Remove the accepted item from candidates
                accepted_id = hits.iloc[0]["candidate_item_id"]
                grp = grp[grp["candidate_item_id"] != accepted_id]

            round_accepts[rnd].append(accepted)

    print(f"\n   {'Round':<10} {'Avg Cumulative Accepts':>25}")
    print(f"   {'─' * 37}")
    for rnd in range(1, max_rounds + 1):
        avg = np.mean(round_accepts[rnd])
        print(f"   Round {rnd:<5} {avg:>25.3f}")

    overall_avg = np.mean(round_accepts[max_rounds])
    print(f"\n   Avg total accepted items after {max_rounds} rounds: {overall_avg:.3f}")
    return round_accepts


# ══════════════════════════════════════════════════════════════════════
#  4️⃣  LATENCY BENCHMARK
# ══════════════════════════════════════════════════════════════════════

def latency_benchmark(model, test_df, feature_cols, n_samples=500):
    """
    Measure per-snapshot inference latency.
    Reports p50 / p95 / p99 in milliseconds.
    """
    print(f"\n{'═' * 60}")
    print(f"  ⏱️  LATENCY BENCHMARK  ({n_samples} snapshots)")
    print(f"{'═' * 60}")

    snap_ids = test_df["snapshot_id"].unique()
    rng = np.random.default_rng(SEED + 1)
    sample_snaps = rng.choice(snap_ids, size=min(n_samples, len(snap_ids)),
                              replace=False)

    latencies_ms = []

    for snap_id in sample_snaps:
        grp = test_df[test_df["snapshot_id"] == snap_id].copy()
        X   = prepare_features_for_eval(grp.copy(), feature_cols)

        t0 = time.perf_counter()
        probs = model.predict_proba(X)[:, 1]
        _ = np.argsort(-probs)[:K]
        t1 = time.perf_counter()

        latencies_ms.append((t1 - t0) * 1000)

    arr = np.array(latencies_ms)
    print(f"\n   p50  : {np.percentile(arr, 50):>8.3f} ms")
    print(f"   p95  : {np.percentile(arr, 95):>8.3f} ms")
    print(f"   p99  : {np.percentile(arr, 99):>8.3f} ms")
    print(f"   mean : {arr.mean():>8.3f} ms")
    print(f"   max  : {arr.max():>8.3f} ms")
    return arr


# ══════════════════════════════════════════════════════════════════════
#  🏁  MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    # ── Load model & feature list ─────────────────────────────────────
    model_path = MODELS / "lightgbm_model.pkl"
    feat_path  = MODELS / "feature_list.json"

    print(f"📦 Loading model from {model_path}")
    model = joblib.load(model_path)

    with open(feat_path) as f:
        feature_cols = json.load(f)
    print(f"📦 Loaded feature list ({len(feature_cols)} features)")

    # ── Load data & temporal split ────────────────────────────────────
    df = load_data()
    train_df, test_df = temporal_split(df)

    # ── 1. Segment analysis ───────────────────────────────────────────
    segment_analysis(model, test_df, feature_cols)

    # ── 2. Cold-start evaluation ──────────────────────────────────────
    cold_start_evaluation(model, test_df, feature_cols)

    # ── 3. Sequential cart simulation ─────────────────────────────────
    sequential_cart_simulation(model, test_df, feature_cols)

    # ── 4. Latency benchmark ─────────────────────────────────────────
    latency_benchmark(model, test_df, feature_cols)

    print("\n🏁 Strategic analysis complete!")


if __name__ == "__main__":
    main()
