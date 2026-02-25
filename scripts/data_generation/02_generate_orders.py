"""
🛒 Step 02 — Generate Orders & Order Items
    orders.csv  ·  order_items.csv
"""

import numpy as np
import pandas as pd
import pathlib
import warnings
from datetime import datetime, timedelta
from tqdm import tqdm

warnings.filterwarnings("ignore")

SEED = 42
rng  = np.random.default_rng(SEED)
np.random.seed(SEED)

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
RAW  = ROOT / "data" / "raw"


# ── Load base tables ─────────────────────────────────────────────────
def load_base_tables():
    users       = pd.read_csv(RAW / "users.csv", parse_dates=["signup_date"])
    restaurants = pd.read_csv(RAW / "restaurants.csv")
    items       = pd.read_csv(RAW / "items.csv")
    print(f"✅ Loaded  users={users.shape}  restaurants={restaurants.shape}  items={items.shape}")
    return users, restaurants, items


# ── Pre-compute helpers ──────────────────────────────────────────────
def build_helpers(items, restaurants):
    rest_items = {}
    for rid, grp in items.groupby("restaurant_id"):
        rest_items[rid] = {cat: sub for cat, sub in grp.groupby("category")}

    city_restaurants = restaurants.groupby("city")["restaurant_id"].apply(list).to_dict()
    rest_price       = restaurants.set_index("restaurant_id")["price_range"].to_dict()

    BUDGET_PRICE_MAP = {
        "budget":  {"low": 0.60, "mid": 0.30, "high": 0.10},
        "mid":     {"low": 0.20, "mid": 0.55, "high": 0.25},
        "premium": {"low": 0.05, "mid": 0.30, "high": 0.65},
    }
    print("✅ Helper structures ready.")
    return rest_items, city_restaurants, rest_price, BUDGET_PRICE_MAP


# ── Timestamp generation ─────────────────────────────────────────────
def generate_timestamps(n: int) -> pd.Series:
    END_TS   = datetime(2026, 2, 24)
    START_TS = END_TS - timedelta(days=365)

    HOUR_WEIGHTS = np.array([
        0.2, 0.1, 0.1, 0.1, 0.1, 0.3,
        0.5, 0.8, 1.2, 1.5, 2.0, 3.5,
        5.0, 4.5, 2.5, 2.0, 2.0, 2.5,
        3.5, 5.0, 5.5, 5.0, 3.0, 1.5,
    ])
    HOUR_WEIGHTS /= HOUR_WEIGHTS.sum()

    days    = rng.integers(0, 365, size=n)
    hours   = rng.choice(24, size=n, p=HOUR_WEIGHTS)
    minutes = rng.integers(0, 60, size=n)
    seconds = rng.integers(0, 60, size=n)

    timestamps = pd.to_datetime([
        START_TS + timedelta(days=int(d), hours=int(h), minutes=int(m), seconds=int(s))
        for d, h, m, s in zip(days, hours, minutes, seconds)
    ])
    return timestamps


# ── Cart simulation ──────────────────────────────────────────────────
def generate_orders(users, restaurants, items, rest_items, city_restaurants,
                    rest_price, BUDGET_PRICE_MAP):
    N_ORDERS = 300_000

    ADDON_BASE  = {"side": 0.60, "drink": 0.55, "dessert": 0.40}
    BUDGET_MULT = {"budget": 0.70, "mid": 1.00, "premium": 1.30}

    user_ids    = users["user_id"].values
    user_cities = users["city"].values
    user_budgets = users["budget_segment"].values
    user_veg    = users["veg_preference"].values

    user_freq   = users["order_frequency"].values.astype(float)
    user_freq_p = user_freq / user_freq.sum()

    def pick_restaurant_for_user(city, budget):
        candidates = city_restaurants.get(city, [])
        if not candidates:
            candidates = restaurants["restaurant_id"].tolist()
        weights = np.array([BUDGET_PRICE_MAP[budget].get(rest_price.get(rid, "mid"), 0.33)
                            for rid in candidates])
        weights /= weights.sum()
        return rng.choice(candidates, p=weights)

    def sample_items_from_cat(rid, cat, n, veg_pref):
        cat_items = rest_items.get(rid, {}).get(cat, None)
        if cat_items is None or len(cat_items) == 0:
            return []
        pool = cat_items
        if veg_pref == "veg":
            veg_pool = pool[pool["veg_or_nonveg"] == "veg"]
            if len(veg_pool) > 0:
                pool = veg_pool
        pop = pool["popularity_score"].values.astype(float)
        if pop.sum() == 0:
            pop = np.ones(len(pop))
        pop /= pop.sum()
        n = min(n, len(pool))
        chosen_idx = rng.choice(len(pool), size=n, replace=False, p=pop)
        return pool.iloc[chosen_idx][["item_id", "price"]].values.tolist()

    # ── Generate timestamps ───────────────────────────────────────
    order_timestamps = generate_timestamps(N_ORDERS)
    print(f"✅ Generated {N_ORDERS:,} timestamps")
    print(f"   Range: {order_timestamps.min()} → {order_timestamps.max()}")

    # ── Main loop ─────────────────────────────────────────────────
    order_rows      = []
    order_item_rows = []
    order_id        = 1

    sampled_user_idx = rng.choice(len(user_ids), size=N_ORDERS, p=user_freq_p)

    for i in tqdm(range(N_ORDERS), desc="Generating orders", mininterval=2):
        uidx   = sampled_user_idx[i]
        uid    = int(user_ids[uidx])
        city   = user_cities[uidx]
        budget = user_budgets[uidx]
        veg    = user_veg[uidx]
        ts     = order_timestamps[i]

        rid = pick_restaurant_for_user(city, budget)

        cart_items = []
        n_mains = rng.choice([1, 2], p=[0.45, 0.55])
        mains   = sample_items_from_cat(rid, "main", n_mains, veg)
        cart_items.extend(mains)

        mult = BUDGET_MULT[budget]
        for addon_cat in ["side", "drink", "dessert"]:
            prob = min(ADDON_BASE[addon_cat] * mult, 0.95)
            if rng.random() < prob:
                addon = sample_items_from_cat(rid, addon_cat, 1, veg)
                cart_items.extend(addon)

        if budget == "premium" and rng.random() < 0.35:
            extra_cat = rng.choice(["drink", "dessert"])
            extra = sample_items_from_cat(rid, extra_cat, 1, veg)
            cart_items.extend(extra)

        if len(cart_items) == 0:
            continue

        total_value = sum(p for _, p in cart_items)
        order_rows.append({
            "order_id":          order_id,
            "user_id":           uid,
            "restaurant_id":     rid,
            "order_timestamp":   ts,
            "city":              city,
            "total_order_value": round(total_value, 2),
        })

        for item_id, item_price in cart_items:
            order_item_rows.append({
                "order_id":   order_id,
                "item_id":    int(item_id),
                "quantity":   1,
                "item_price": round(item_price, 2),
            })

        order_id += 1

    print(f"\n✅ Generated {len(order_rows):,} orders with {len(order_item_rows):,} line items")
    return pd.DataFrame(order_rows), pd.DataFrame(order_item_rows)


def save_and_update_users(orders, order_items, users, items):
    """Save CSVs and update users with total_orders."""
    orders.to_csv(RAW / "orders.csv", index=False)
    order_items.to_csv(RAW / "order_items.csv", index=False)

    user_order_counts = orders.groupby("user_id").size().rename("total_orders")
    users = users.set_index("user_id")
    users["total_orders"] = user_order_counts
    users["total_orders"] = users["total_orders"].fillna(0).astype(int)
    users = users.reset_index()
    users.to_csv(RAW / "users.csv", index=False)

    print(f"✅ orders.csv       →  {orders.shape}")
    print(f"✅ order_items.csv  →  {order_items.shape}")
    print(f"✅ users.csv updated with total_orders")
    print(f"\n   Cold-start users (< 3 orders): {(users['total_orders'] < 3).sum():,}")

    return users


def sanity_checks(orders, order_items, items):
    """Print sanity checks."""
    items_per_order = order_items.groupby("order_id").size()
    print("=" * 60)
    print("ORDERS")
    print(f"  Shape               : {orders.shape}")
    print(f"  Avg order value     : ₹{orders['total_order_value'].mean():.2f}")
    print(f"  Median order value  : ₹{orders['total_order_value'].median():.2f}")
    print()
    print("ORDER ITEMS")
    print(f"  Shape               : {order_items.shape}")
    print(f"  Avg items per order : {items_per_order.mean():.2f}")
    print(f"  Max items per order : {items_per_order.max()}")
    print()

    orders_ts = pd.to_datetime(orders["order_timestamp"])
    hour_counts = orders_ts.dt.hour.value_counts().sort_index()
    peak_lunch  = hour_counts.loc[12:14].sum()
    peak_dinner = hour_counts.loc[19:21].sum()
    print(f"  Peak lunch  (12-14) : {peak_lunch:,} orders ({peak_lunch/len(orders):.1%})")
    print(f"  Peak dinner (19-21) : {peak_dinner:,} orders ({peak_dinner/len(orders):.1%})")

    oi_with_cat = order_items.merge(items[["item_id", "category"]], on="item_id", how="left")
    print(f"\n  Category distribution in carts:")
    print(f"  {oi_with_cat['category'].value_counts(normalize=True).round(3).to_dict()}")
    print("=" * 60)


def main():
    users, restaurants, items = load_base_tables()
    rest_items, city_restaurants, rest_price, BUDGET_PRICE_MAP = build_helpers(items, restaurants)
    orders, order_items = generate_orders(
        users, restaurants, items, rest_items, city_restaurants, rest_price, BUDGET_PRICE_MAP
    )
    users = save_and_update_users(orders, order_items, users, items)
    sanity_checks(orders, order_items, items)


if __name__ == "__main__":
    main()
