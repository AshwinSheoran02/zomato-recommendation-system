"""
🧠 Step 03 — Build Training Table
    training_rows.csv  ·  baseline_top10.csv
"""

import os
import numpy as np
import pandas as pd
import pathlib
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

SEED = 42
rng  = np.random.default_rng(SEED)
np.random.seed(SEED)

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW  = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

N_CANDIDATES = 40  # target candidates per snapshot (30-50 range)


# ── Load ──────────────────────────────────────────────────────────────
def load_all():
    users       = pd.read_csv(RAW / "users.csv")
    restaurants = pd.read_csv(RAW / "restaurants.csv")
    items       = pd.read_csv(RAW / "items.csv")
    orders      = pd.read_csv(RAW / "orders.csv", parse_dates=["order_timestamp"])
    order_items = pd.read_csv(RAW / "order_items.csv")

    print("✅ Loaded all raw tables")
    print(f"   users={users.shape}  restaurants={restaurants.shape}  items={items.shape}")
    print(f"   orders={orders.shape}  order_items={order_items.shape}")
    return users, restaurants, items, orders, order_items


# ── Lookups ───────────────────────────────────────────────────────────
def build_lookups(users, items, order_items):
    item_meta = items.set_index("item_id")[
        ["restaurant_id", "category", "price", "popularity_score", "veg_or_nonveg"]
    ].to_dict(orient="index")

    rest_item_ids = items.groupby("restaurant_id")["item_id"].apply(list).to_dict()

    baseline_top10 = (
        items.sort_values("popularity_score", ascending=False)
             .groupby("restaurant_id")["item_id"]
             .apply(lambda x: x.head(10).tolist())
             .to_dict()
    )

    user_veg_map = users.set_index("user_id")["veg_preference"].to_dict()

    oi_enriched = order_items.merge(
        items[["item_id", "category", "price", "popularity_score", "veg_or_nonveg"]],
        on="item_id", how="left",
    )

    print(f"✅ Lookups ready")
    print(f"   Baseline top-10 computed for {len(baseline_top10):,} restaurants")
    return item_meta, rest_item_ids, baseline_top10, user_veg_map, oi_enriched


# ── Snapshot helpers ──────────────────────────────────────────────────
def make_snapshot_features(snapshot_items, order_row, user_veg, item_meta):
    cats_in_cart = set()
    cart_value   = 0.0
    for iid in snapshot_items:
        meta = item_meta.get(iid, {})
        cart_value += meta.get("price", 0)
        cats_in_cart.add(meta.get("category", ""))
    return {
        "order_id":        int(order_row["order_id"]),
        "order_timestamp": order_row["order_timestamp"],
        "user_id":         int(order_row["user_id"]),
        "restaurant_id":   int(order_row["restaurant_id"]),
        "cart_value":      round(cart_value, 2),
        "cart_item_count": len(snapshot_items),
        "has_drink":       int("drink" in cats_in_cart),
        "has_dessert":     int("dessert" in cats_in_cart),
        "hour_of_day":     order_row["order_timestamp"].hour,
        "weekday":         order_row["order_timestamp"].weekday(),
        "_snapshot_ids":   set(snapshot_items),
    }


def generate_candidates(rid, snapshot_ids, final_ids, user_veg,
                         rest_item_ids, item_meta, n=N_CANDIDATES):
    all_items = rest_item_ids.get(rid, [])
    pool = [iid for iid in all_items if iid not in snapshot_ids]
    if len(pool) == 0:
        return []

    pops = np.array([item_meta[iid]["popularity_score"] for iid in pool])
    if pops.sum() == 0:
        pops = np.ones(len(pops))
    pops /= pops.sum()

    n_sample = min(n, len(pool))
    chosen   = rng.choice(pool, size=n_sample, replace=False, p=pops)

    positives_not_in_snapshot = final_ids - snapshot_ids
    for pos_id in positives_not_in_snapshot:
        if pos_id not in chosen and pos_id in pool:
            chosen = np.append(chosen, pos_id)

    rows = []
    for cid in chosen:
        meta = item_meta.get(cid, {})
        veg_match = int(
            (user_veg == "veg"     and meta.get("veg_or_nonveg") == "veg") or
            (user_veg == "non-veg" and meta.get("veg_or_nonveg") == "non-veg") or
            (user_veg == "mixed")
        )
        rows.append({
            "candidate_item_id":     int(cid),
            "candidate_price":       meta.get("price", 0),
            "candidate_category":    meta.get("category", ""),
            "candidate_popularity":  meta.get("popularity_score", 0),
            "matches_user_veg_pref": veg_match,
            "label":                 int(cid in final_ids),
        })
    return rows


# ── Main training-row loop ────────────────────────────────────────────
def build_training_rows(orders, oi_enriched, user_veg_map, rest_item_ids, item_meta):
    training_rows = []
    snapshot_counter = 0

    SAMPLE_ORDERS  = min(12_000, len(orders))
    sampled_orders = orders.sample(n=SAMPLE_ORDERS, random_state=SEED)

    for _, orow in tqdm(sampled_orders.iterrows(), total=SAMPLE_ORDERS,
                         desc="Building training rows", mininterval=2):
        oid = orow["order_id"]
        rid = orow["restaurant_id"]
        uid = orow["user_id"]
        user_veg = user_veg_map.get(uid, "mixed")

        order_item_list = oi_enriched[oi_enriched["order_id"] == oid].sort_values(
            "category", key=lambda s: s.map({"main": 0, "side": 1, "drink": 2, "dessert": 3})
        )["item_id"].tolist()

        if len(order_item_list) < 2:
            continue

        final_ids = set(order_item_list)

        # Snapshot 1: after first item
        snapshot_counter += 1
        snap1 = order_item_list[:1]
        feat1 = make_snapshot_features(snap1, orow, user_veg, item_meta)
        feat1["snapshot_id"] = snapshot_counter
        cands1 = generate_candidates(rid, feat1["_snapshot_ids"], final_ids, user_veg,
                                      rest_item_ids, item_meta)
        for c in cands1:
            row = {k: v for k, v in feat1.items() if not k.startswith("_")}
            row.update(c)
            training_rows.append(row)

        # Snapshot 2: after second item
        snapshot_counter += 1
        snap2 = order_item_list[:2]
        feat2 = make_snapshot_features(snap2, orow, user_veg, item_meta)
        feat2["snapshot_id"] = snapshot_counter
        cands2 = generate_candidates(rid, feat2["_snapshot_ids"], final_ids, user_veg,
                                      rest_item_ids, item_meta)
        for c in cands2:
            row = {k: v for k, v in feat2.items() if not k.startswith("_")}
            row.update(c)
            training_rows.append(row)

    print(f"\n✅ Generated {len(training_rows):,} training rows")
    return training_rows, sampled_orders


def save_training(training_rows):
    training = pd.DataFrame(training_rows)

    COL_ORDER = [
        "snapshot_id", "order_id", "order_timestamp",
        "user_id", "restaurant_id", "cart_value", "cart_item_count",
        "has_drink", "has_dessert", "hour_of_day", "weekday",
        "candidate_item_id", "candidate_price", "candidate_category",
        "candidate_popularity", "matches_user_veg_pref", "label",
    ]
    training = training[COL_ORDER]
    training.to_csv(PROC / "training_rows.csv", index=False)

    print(f"✅ training_rows.csv  →  {training.shape}")
    print(f"\n📊 Label distribution:")
    print(training["label"].value_counts())
    print(f"\n   Positive rate: {training['label'].mean():.2%}")
    return training


def save_baseline(baseline_top10, item_meta):
    baseline_rows = []
    for rid, top_items in baseline_top10.items():
        for rank, iid in enumerate(top_items, 1):
            meta = item_meta.get(iid, {})
            baseline_rows.append({
                "restaurant_id":    rid,
                "rank":             rank,
                "item_id":          iid,
                "category":         meta.get("category", ""),
                "price":            meta.get("price", 0),
                "popularity_score": meta.get("popularity_score", 0),
            })
    baseline = pd.DataFrame(baseline_rows)
    baseline.to_csv(PROC / "baseline_top10.csv", index=False)
    print(f"✅ baseline_top10.csv  →  {baseline.shape}")


def temporal_split_preview(orders):
    orders_sorted = orders.sort_values("order_timestamp")
    split_idx  = int(len(orders_sorted) * 0.80)
    split_date = orders_sorted.iloc[split_idx]["order_timestamp"]

    train_orders = orders_sorted.iloc[:split_idx]
    test_orders  = orders_sorted.iloc[split_idx:]

    print(f"\n📅 Temporal split date : {split_date}")
    print(f"   Train orders        : {len(train_orders):,}  ({len(train_orders)/len(orders):.0%})")
    print(f"   Test  orders        : {len(test_orders):,}  ({len(test_orders)/len(orders):.0%})")

    user_orders_before = train_orders.groupby("user_id").size()
    test_user_ids = test_orders["user_id"].unique()
    cold_start = sum(1 for uid in test_user_ids if user_orders_before.get(uid, 0) < 3)
    print(f"\n   Cold-start users in test (< 3 orders before split):")
    print(f"   {cold_start:,} users ({cold_start/len(test_user_ids):.1%} of test users)")
    return split_date, cold_start


def final_summary(training, split_date, cold_start):
    print("\n" + "=" * 65)
    print("  📂 GENERATED DATASET SUMMARY")
    print("=" * 65)

    files = {
        "data/raw/users.csv":               RAW / "users.csv",
        "data/raw/restaurants.csv":         RAW / "restaurants.csv",
        "data/raw/items.csv":               RAW / "items.csv",
        "data/raw/orders.csv":              RAW / "orders.csv",
        "data/raw/order_items.csv":         RAW / "order_items.csv",
        "data/processed/training_rows.csv": PROC / "training_rows.csv",
        "data/processed/baseline_top10.csv": PROC / "baseline_top10.csv",
    }

    for label, path in files.items():
        if path.exists():
            size_mb = os.path.getsize(path) / (1024 * 1024)
            df_temp = pd.read_csv(path, nrows=0)
            n_cols  = len(df_temp.columns)
            n_rows  = sum(1 for _ in open(path)) - 1
            print(f"  ✅ {label:<42s}  {n_rows:>10,} rows × {n_cols} cols  ({size_mb:.1f} MB)")
        else:
            print(f"  ❌ {label:<42s}  NOT FOUND")

    print("=" * 65)
    print(f"\n  🎯 Random seed: {SEED}")
    print(f"  📅 Temporal split: {split_date}")
    print(f"  🧊 Cold-start users: {cold_start:,}")
    print(f"  📈 Training label=1 rate: {training['label'].mean():.2%}")
    print(f"\n  🏁 All datasets ready for model training!")


def main():
    users, restaurants, items, orders, order_items = load_all()
    item_meta, rest_item_ids, baseline_top10, user_veg_map, oi_enriched = \
        build_lookups(users, items, order_items)

    training_rows, sampled_orders = build_training_rows(
        orders, oi_enriched, user_veg_map, rest_item_ids, item_meta
    )
    training = save_training(training_rows)
    save_baseline(baseline_top10, item_meta)
    split_date, cold_start = temporal_split_preview(orders)
    final_summary(training, split_date, cold_start)


if __name__ == "__main__":
    main()
