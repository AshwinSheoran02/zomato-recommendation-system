"""
🏋️ Step 01 — Train LightGBM Model
====================================
  1. Load training_rows.csv
  2. Temporal train/test split (80/20 by date)
  3. Prepare features (label-encode, drop IDs)
  4. Train LightGBM classifier
  5. Save model  → models/lightgbm_model.pkl
  6. Save feature list → models/feature_list.json

Usage:
    python scripts/01_train_model.py
"""

import json
import numpy as np
import pandas as pd
import pathlib
import warnings
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import joblib

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

ROOT = pathlib.Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
#  SHARED HELPERS  (same logic used by 02 & 03 scripts)
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


def temporal_split(df, ratio=0.80):
    """
    Split by date so that test = last (1-ratio) of calendar days.
    All rows from a given snapshot stay together → no data leakage.
    """
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


def prepare_features(train_df, test_df):
    """
    • Label-encode candidate_category
    • Drop ID / timestamp / grouping columns
    • Return X_train, y_train, X_test, y_test, feature_cols
    """
    le = LabelEncoder()
    train_df["candidate_category"] = le.fit_transform(train_df["candidate_category"])
    test_df["candidate_category"]  = le.transform(test_df["candidate_category"])

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
#  🏁  MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    # 1. Load
    df = load_data()

    # 2. Temporal split
    train_df, test_df = temporal_split(df)

    # 3. Feature preparation
    X_train, y_train, X_test, y_test, feature_cols = \
        prepare_features(train_df, test_df)

    # 4. Train LightGBM
    model = train_lgbm(X_train, y_train)

    # 5. Save model artifact
    model_path = MODELS / "lightgbm_model.pkl"
    joblib.dump(model, model_path)
    print(f"\n💾 Model saved → {model_path}")

    # 6. Save feature list
    feat_path = MODELS / "feature_list.json"
    with open(feat_path, "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"💾 Feature list saved → {feat_path}")

    print("\n🏁 Training complete!")


if __name__ == "__main__":
    main()
