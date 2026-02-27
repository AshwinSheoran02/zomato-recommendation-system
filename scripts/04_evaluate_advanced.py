"""
📊 Step 04 — Advanced Evaluation: Old vs New Architecture
==========================================================
  Head-to-head comparison of 3 architectures:

    1. Baseline      — Popularity ranking (no model)
    2. LightGBM      — LightGBM predict_proba ranking
    3. Advanced       — Agentic Scorer + GraphRAG + Causal Debiasing + CRC

  Metrics:
    • Precision@10, Recall@10, Acceptance Rate
    • Average Order Value (AOV) with reco revenue
    • Category Diversity (unique categories in top-10)
    • Long-Tail Coverage (% from bottom-50% popularity)
    • Safety Violation Rate (non-veg for veg users)
    • Debiasing Impact (Gini of recommended item popularity)

  Charts saved to assets/figures/:
    • advanced_precision_recall.png
    • advanced_aov_comparison.png
    • advanced_diversity_longtail.png
    • advanced_safety_report.png
    • advanced_debiasing_gini.png

Usage:
    python scripts/04_evaluate_advanced.py
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
from matplotlib.patches import FancyBboxPatch

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

ROOT   = pathlib.Path(__file__).resolve().parent.parent
PROC   = ROOT / "data" / "processed"
RAW    = ROOT / "data" / "raw"
MODELS = ROOT / "models"
FIGS   = ROOT / "assets" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

K = 10

# ── Rich color palette ────────────────────────────────────────────
COLORS = {
    "baseline": "#e74c3c",    # Red
    "lightgbm": "#f39c12",    # Amber
    "advanced": "#2ecc71",    # Green
}
ARCH_LABELS = ["Baseline", "LightGBM", "Advanced"]
ARCH_COLORS = [COLORS["baseline"], COLORS["lightgbm"], COLORS["advanced"]]


# ══════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════

def load_all():
    """Load training data, model, and advanced scores."""
    # Training data
    df = pd.read_csv(PROC / "training_rows.csv", parse_dates=["order_timestamp"])
    print(f"✅ Loaded training_rows.csv → {df.shape}")

    # LightGBM model
    model = joblib.load(MODELS / "lightgbm_model.pkl")
    with open(MODELS / "feature_list.json") as f:
        feature_cols = json.load(f)
    print(f"✅ Loaded LightGBM model ({len(feature_cols)} features)")

    # Advanced scores
    adv_scores = pd.read_csv(PROC / "advanced_scores.csv")
    print(f"✅ Loaded advanced_scores.csv → {adv_scores.shape}")

    # Items for metadata
    items = pd.read_csv(RAW / "items.csv")
    users = pd.read_csv(RAW / "users.csv")

    return df, model, feature_cols, adv_scores, items, users


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
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df["candidate_category"] = le.fit_transform(df["candidate_category"])
    X = df[feature_cols]
    return X


# ══════════════════════════════════════════════════════════════════════
#  PER-SNAPSHOT EVALUATION
# ══════════════════════════════════════════════════════════════════════

def evaluate_architecture(test_df, score_col, arch_name, items_df, users_df, k=K):
    """
    Evaluate a single architecture by ranking candidates per snapshot.

    Returns a DataFrame of per-snapshot metrics.
    """
    item_veg  = items_df.set_index("item_id")["veg_or_nonveg"].to_dict()
    item_cat  = items_df.set_index("item_id")["category"].to_dict()
    user_veg  = users_df.set_index("user_id")["veg_preference"].to_dict()
    pop_median = items_df["popularity_score"].median()

    records = []
    for snap_id, grp in test_df.groupby("snapshot_id"):
        ranked = grp.sort_values(score_col, ascending=False)
        top_k  = ranked.head(k)

        n_relevant = grp["label"].sum()
        hits       = top_k["label"].sum()

        # Category diversity
        categories_in_topk = set()
        for _, row in top_k.iterrows():
            cid = int(row["candidate_item_id"])
            categories_in_topk.add(item_cat.get(cid, row.get("candidate_category", "")))

        # Long-tail coverage
        longtail_count = sum(
            1 for _, row in top_k.iterrows()
            if row["candidate_popularity"] < pop_median
        )

        # Safety violations
        uid = int(grp["user_id"].iloc[0])
        u_veg = user_veg.get(uid, "mixed")
        safety_violations = 0
        if u_veg == "veg":
            for _, row in top_k.iterrows():
                cid = int(row["candidate_item_id"])
                if item_veg.get(cid, "non-veg") == "non-veg":
                    safety_violations += 1

        records.append({
            "snapshot_id":     snap_id,
            "precision_at_k":  hits / k,
            "recall_at_k":     (hits / n_relevant) if n_relevant > 0 else 0.0,
            "any_hit":         int(hits > 0),
            "hit_revenue":     top_k.loc[top_k["label"] == 1, "candidate_price"].sum(),
            "n_hits":          hits,
            "cart_value":      grp["cart_value"].iloc[0],
            "cart_item_count": grp["cart_item_count"].iloc[0],
            "n_categories":    len(categories_in_topk),
            "longtail_count":  longtail_count,
            "safety_violations": safety_violations,
            "is_veg_user":     int(u_veg == "veg"),
            # For Gini computation: collect top-k popularities
            "topk_pops":       top_k["candidate_popularity"].tolist(),
        })

    res = pd.DataFrame(records)
    return res


def compute_gini(values):
    """Compute Gini coefficient of a distribution."""
    values = np.array(values, dtype=float)
    if len(values) == 0 or values.sum() == 0:
        return 0.0
    values = np.sort(values)
    n = len(values)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * values) - (n + 1) * np.sum(values)) / (n * np.sum(values))


# ══════════════════════════════════════════════════════════════════════
#  AGGREGATE METRICS
# ══════════════════════════════════════════════════════════════════════

def aggregate_metrics(res, arch_name):
    """Compute aggregate metrics from per-snapshot results."""
    prec      = res["precision_at_k"].mean()
    rec       = res["recall_at_k"].mean()
    acc       = res["any_hit"].mean()
    aov       = (res["cart_value"] + res["hit_revenue"]).mean()
    items_per = (res["cart_item_count"] + res["n_hits"]).mean()
    diversity = res["n_categories"].mean()
    longtail  = res["longtail_count"].mean() / K  # fraction

    # Safety: violation rate among veg users
    veg_mask = res["is_veg_user"] == 1
    if veg_mask.sum() > 0:
        safety_violation_rate = res.loc[veg_mask, "safety_violations"].mean() / K
    else:
        safety_violation_rate = 0.0

    # Gini of recommended item popularities
    all_pops = []
    for pops in res["topk_pops"]:
        all_pops.extend(pops)
    gini = compute_gini(all_pops)

    metrics = {
        "Precision@10":              prec,
        "Recall@10":                 rec,
        "Acceptance Rate":           acc,
        "Avg Order Value (₹)":       aov,
        "Avg Items / Order":         items_per,
        "Category Diversity":        diversity,
        "Long-Tail Coverage":        longtail,
        "Safety Violation Rate":     safety_violation_rate,
        "Popularity Gini":           gini,
    }

    print(f"\n{'─' * 55}")
    print(f"  📊  {arch_name}")
    print(f"{'─' * 55}")
    for metric, val in metrics.items():
        if "Rate" in metric or "Coverage" in metric:
            print(f"   {metric:<28s}: {val:.2%}")
        elif "₹" in metric:
            print(f"   {metric:<28s}: ₹{val:.2f}")
        else:
            print(f"   {metric:<28s}: {val:.4f}")

    return metrics


# ══════════════════════════════════════════════════════════════════════
#  COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════

def build_comparison_table(all_metrics):
    """Build a comparison DataFrame from all architectures."""
    comp = pd.DataFrame(all_metrics)
    comp.index.name = "Metric"

    # Compute lifts vs baseline
    for col in comp.columns:
        if col != "Baseline":
            comp[f"{col} vs Baseline"] = (
                (comp[col] - comp["Baseline"]) / comp["Baseline"].abs() * 100
            ).round(2)

    print("\n" + "=" * 90)
    print("  📊  FULL COMPARISON — BASELINE vs LIGHTGBM vs ADVANCED")
    print("=" * 90)
    print(comp.to_string(float_format=lambda x: f"{x:.4f}"))
    print("=" * 90)
    return comp


# ══════════════════════════════════════════════════════════════════════
#  CHARTS
# ══════════════════════════════════════════════════════════════════════

def setup_plot_style():
    """Configure matplotlib for premium-looking charts."""
    plt.rcParams.update({
        "figure.facecolor": "#1a1a2e",
        "axes.facecolor":   "#16213e",
        "text.color":       "#e8e8e8",
        "axes.labelcolor":  "#e8e8e8",
        "xtick.color":      "#e8e8e8",
        "ytick.color":      "#e8e8e8",
        "axes.edgecolor":   "#333366",
        "grid.color":       "#333366",
        "grid.alpha":       0.3,
        "font.family":      "sans-serif",
        "font.size":        12,
    })


def plot_precision_recall(all_metrics):
    """3-way bar chart for Precision and Recall."""
    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    metrics = ["Precision@10", "Recall@10", "Acceptance Rate"]
    for ax, metric in zip(axes, metrics):
        values = [all_metrics[arch][metric] for arch in ARCH_LABELS]
        bars = ax.bar(ARCH_LABELS, values, color=ARCH_COLORS, width=0.55,
                      edgecolor="#ffffff22", linewidth=0.5)
        for b in bars:
            ht = b.get_height()
            fmt = ".2%" if "Rate" in metric else ".4f"
            ax.text(b.get_x() + b.get_width() / 2, ht + max(values) * 0.02,
                    f"{ht:{fmt}}", ha="center", va="bottom", fontsize=11,
                    fontweight="bold", color="#ffffff")
        ax.set_title(metric, fontsize=14, fontweight="bold", color="#ffffff")
        ax.set_ylim(0, max(values) * 1.25)
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle("Recommendation Quality — 3 Architecture Comparison",
                 fontsize=16, fontweight="bold", color="#ffffff", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGS / "advanced_precision_recall.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print("   📈 Saved advanced_precision_recall.png")


def plot_aov_comparison(all_metrics):
    """AOV comparison chart."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    values = [all_metrics[arch]["Avg Order Value (₹)"] for arch in ARCH_LABELS]
    bars = ax.bar(ARCH_LABELS, values, color=ARCH_COLORS, width=0.50,
                  edgecolor="#ffffff22", linewidth=0.5)
    for b in bars:
        ht = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, ht + max(values) * 0.02,
                f"₹{ht:.1f}", ha="center", va="bottom", fontsize=12,
                fontweight="bold", color="#ffffff")

    ax.set_title("Average Order Value with Recommendations",
                 fontsize=14, fontweight="bold", color="#ffffff")
    ax.set_ylabel("₹ (INR)", fontsize=12)
    ax.set_ylim(0, max(values) * 1.20)
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(FIGS / "advanced_aov_comparison.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print("   📈 Saved advanced_aov_comparison.png")


def plot_diversity_longtail(all_metrics):
    """Diversity and long-tail coverage comparison."""
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Category Diversity
    ax = axes[0]
    values = [all_metrics[arch]["Category Diversity"] for arch in ARCH_LABELS]
    bars = ax.bar(ARCH_LABELS, values, color=ARCH_COLORS, width=0.50,
                  edgecolor="#ffffff22", linewidth=0.5)
    for b in bars:
        ht = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, ht + max(values) * 0.02,
                f"{ht:.2f}", ha="center", va="bottom", fontsize=11,
                fontweight="bold", color="#ffffff")
    ax.set_title("Category Diversity (unique categories in top-10)",
                 fontsize=13, fontweight="bold", color="#ffffff")
    ax.set_ylim(0, max(values) * 1.25)
    ax.grid(axis="y", alpha=0.2)

    # Long-Tail Coverage
    ax = axes[1]
    values = [all_metrics[arch]["Long-Tail Coverage"] for arch in ARCH_LABELS]
    bars = ax.bar(ARCH_LABELS, values, color=ARCH_COLORS, width=0.50,
                  edgecolor="#ffffff22", linewidth=0.5)
    for b in bars:
        ht = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, ht + max(values) * 0.02,
                f"{ht:.1%}", ha="center", va="bottom", fontsize=11,
                fontweight="bold", color="#ffffff")
    ax.set_title("Long-Tail Coverage (below-median popularity items)",
                 fontsize=13, fontweight="bold", color="#ffffff")
    ax.set_ylim(0, max(max(values) * 1.25, 0.1))
    ax.grid(axis="y", alpha=0.2)

    fig.suptitle("Fairness & Discovery Metrics",
                 fontsize=16, fontweight="bold", color="#ffffff", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGS / "advanced_diversity_longtail.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print("   📈 Saved advanced_diversity_longtail.png")


def plot_safety_report(all_metrics):
    """Safety violation rate comparison."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    values = [all_metrics[arch]["Safety Violation Rate"] for arch in ARCH_LABELS]
    bars = ax.bar(ARCH_LABELS, values, color=ARCH_COLORS, width=0.50,
                  edgecolor="#ffffff22", linewidth=0.5)
    for b in bars:
        ht = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, ht + max(max(values), 0.001) * 0.05,
                f"{ht:.2%}", ha="center", va="bottom", fontsize=12,
                fontweight="bold", color="#ffffff")

    # Draw safety threshold line
    ax.axhline(y=0.005, color="#ff6b6b", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.text(2.5, 0.005, "CRC Threshold (0.5%)", color="#ff6b6b",
            fontsize=10, va="bottom", ha="right")

    ax.set_title("Dietary Safety Violation Rate (Veg Users → Non-Veg Recos)",
                 fontsize=13, fontweight="bold", color="#ffffff")
    ax.set_ylabel("Violation Rate", fontsize=12)
    ax.set_ylim(0, max(max(values) * 1.5, 0.01))
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(FIGS / "advanced_safety_report.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print("   📈 Saved advanced_safety_report.png")


def plot_debiasing_gini(all_metrics):
    """Popularity distribution Gini comparison."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    values = [all_metrics[arch]["Popularity Gini"] for arch in ARCH_LABELS]
    bars = ax.bar(ARCH_LABELS, values, color=ARCH_COLORS, width=0.50,
                  edgecolor="#ffffff22", linewidth=0.5)
    for b in bars:
        ht = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, ht + max(values) * 0.02,
                f"{ht:.4f}", ha="center", va="bottom", fontsize=12,
                fontweight="bold", color="#ffffff")

    ax.set_title("Popularity Gini Coefficient (Lower = More Equitable Distribution)",
                 fontsize=13, fontweight="bold", color="#ffffff")
    ax.set_ylabel("Gini Coefficient", fontsize=12)
    ax.set_ylim(0, max(values) * 1.25)
    ax.grid(axis="y", alpha=0.2)

    # Add annotation
    ax.annotate("↓ Lower Gini indicates\n   successful debiasing",
                xy=(2, values[2]), xytext=(2.3, values[0] * 0.9),
                fontsize=10, color="#2ecc71",
                arrowprops=dict(arrowstyle="->", color="#2ecc71", lw=1.5))

    fig.tight_layout()
    fig.savefig(FIGS / "advanced_debiasing_gini.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print("   📈 Saved advanced_debiasing_gini.png")


# ══════════════════════════════════════════════════════════════════════
#  🏁  MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  📊 ADVANCED EVALUATION — OLD vs NEW ARCHITECTURE")
    print("=" * 70)

    # 1. Load data
    df, model, feature_cols, adv_scores, items, users = load_all()

    # 2. Temporal split
    train_df, test_df = temporal_split(df)

    # 3. Prepare LightGBM features
    test_df_lgb = test_df.copy()
    X_test = prepare_features_for_eval(test_df_lgb, feature_cols)
    test_df_lgb["lgbm_score"] = model.predict_proba(X_test)[:, 1]

    # 4. Merge advanced scores into test_df
    adv_score_map = adv_scores.set_index(
        ["snapshot_id", "candidate_item_id"]
    )["agentic_score"].to_dict()

    test_df_eval = test_df_lgb.copy()
    test_df_eval["advanced_score"] = test_df_eval.apply(
        lambda r: adv_score_map.get(
            (r["snapshot_id"], r["candidate_item_id"]), 0.0
        ), axis=1
    )

    # 5. Evaluate all 3 architectures
    print("\n" + "═" * 70)
    print("  🏗️  EVALUATING 3 ARCHITECTURES")
    print("═" * 70)

    bl_res = evaluate_architecture(
        test_df_eval, "candidate_popularity", "BASELINE (Popularity)", items, users
    )
    lgb_res = evaluate_architecture(
        test_df_eval, "lgbm_score", "LIGHTGBM (Old Architecture)", items, users
    )
    adv_res = evaluate_architecture(
        test_df_eval, "advanced_score", "ADVANCED (New Architecture)", items, users
    )

    # 6. Aggregate metrics
    bl_metrics  = aggregate_metrics(bl_res,  "BASELINE (Popularity Ranking)")
    lgb_metrics = aggregate_metrics(lgb_res, "LIGHTGBM (Old Architecture)")
    adv_metrics = aggregate_metrics(adv_res, "ADVANCED (Agentic + Graph + Causal)")

    all_metrics = {
        "Baseline":  bl_metrics,
        "LightGBM":  lgb_metrics,
        "Advanced":  adv_metrics,
    }

    # 7. Comparison table
    comp = build_comparison_table(all_metrics)

    # 8. Generate charts
    print(f"\n🎨 Generating comparison charts → assets/figures/")
    plot_precision_recall(all_metrics)
    plot_aov_comparison(all_metrics)
    plot_diversity_longtail(all_metrics)
    plot_safety_report(all_metrics)
    plot_debiasing_gini(all_metrics)

    print("\n✅ All 5 charts saved to assets/figures/")
    print("\n🏁 Advanced evaluation complete!")


if __name__ == "__main__":
    main()
