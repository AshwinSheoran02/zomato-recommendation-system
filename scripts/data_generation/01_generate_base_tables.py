"""
📋 Step 01 — Generate Base Tables
    users.csv  ·  restaurants.csv  ·  items.csv
"""

import numpy as np
import pandas as pd
import pathlib
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

SEED = 42
rng  = np.random.default_rng(SEED)
np.random.seed(SEED)

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
RAW  = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)


def generate_users() -> pd.DataFrame:
    """Generate users.csv  (30 000 rows)."""
    N_USERS = 30_000

    CITIES = ["Delhi", "Mumbai", "Bangalore", "Hyderabad", "Chennai",
              "Kolkata", "Pune", "Jaipur", "Lucknow", "Ahmedabad"]

    BUDGET_PROBS  = [0.50, 0.35, 0.15]
    BUDGET_LABELS = ["budget", "mid", "premium"]

    VEG_PROBS  = [0.40, 0.50, 0.10]
    VEG_LABELS = ["veg", "non-veg", "mixed"]

    AOV_RANGES = {"budget": (200, 400), "mid": (350, 700), "premium": (600, 1500)}

    budget_segments = rng.choice(BUDGET_LABELS, size=N_USERS, p=BUDGET_PROBS)
    veg_preferences = rng.choice(VEG_LABELS,    size=N_USERS, p=VEG_PROBS)
    cities          = rng.choice(CITIES,         size=N_USERS)

    end_date   = datetime(2026, 2, 24)
    start_date = end_date - timedelta(days=730)
    signup_dates = pd.to_datetime(
        rng.integers(int(start_date.timestamp()), int(end_date.timestamp()), size=N_USERS),
        unit="s",
    )

    avg_order_values = np.array([
        round(rng.uniform(*AOV_RANGES[b]), 2) for b in budget_segments
    ])

    freq_map = {"budget": (1, 4), "mid": (2, 7), "premium": (3, 10)}
    order_frequencies = np.array([
        round(rng.uniform(*freq_map[b]), 2) for b in budget_segments
    ])

    users = pd.DataFrame({
        "user_id":         np.arange(1, N_USERS + 1),
        "city":            cities,
        "signup_date":     signup_dates,
        "total_orders":    0,
        "avg_order_value": avg_order_values,
        "order_frequency": order_frequencies,
        "budget_segment":  budget_segments,
        "veg_preference":  veg_preferences,
    })

    users.to_csv(RAW / "users.csv", index=False)
    print(f"✅ users.csv  →  {users.shape}")
    return users


def generate_restaurants() -> pd.DataFrame:
    """Generate restaurants.csv  (1 000 rows)."""
    N_RESTAURANTS = 1_000

    CITIES = ["Delhi", "Mumbai", "Bangalore", "Hyderabad", "Chennai",
              "Kolkata", "Pune", "Jaipur", "Lucknow", "Ahmedabad"]

    CUISINES     = ["North Indian", "South Indian", "Chinese", "Fast Food",
                    "Italian", "Biryani", "Desserts"]
    PRICE_RANGES = ["low", "mid", "high"]
    PRICE_PROBS  = [0.35, 0.40, 0.25]

    restaurants = pd.DataFrame({
        "restaurant_id": np.arange(1, N_RESTAURANTS + 1),
        "city":          rng.choice(CITIES,       size=N_RESTAURANTS),
        "cuisine_type":  rng.choice(CUISINES,     size=N_RESTAURANTS),
        "price_range":   rng.choice(PRICE_RANGES, size=N_RESTAURANTS, p=PRICE_PROBS),
        "rating":        np.round(rng.uniform(3.0, 4.8, size=N_RESTAURANTS), 1),
        "is_chain":      rng.choice([0, 1], size=N_RESTAURANTS, p=[0.70, 0.30]),
    })

    restaurants.to_csv(RAW / "restaurants.csv", index=False)
    print(f"✅ restaurants.csv  →  {restaurants.shape}")
    return restaurants


def generate_items(restaurants: pd.DataFrame) -> pd.DataFrame:
    """Generate items.csv  (~30k–60k rows)."""

    ITEM_NAMES = {
        "North Indian": {
            "main":    ["Butter Chicken", "Dal Makhani", "Paneer Tikka Masala",
                        "Rogan Josh", "Chole Bhature", "Kadhai Paneer",
                        "Shahi Paneer", "Malai Kofta", "Aloo Gobi", "Rajma Chawal",
                        "Tandoori Chicken", "Chicken Tikka", "Naan Combo",
                        "Mutton Curry", "Palak Paneer"],
            "side":    ["Raita", "Papad", "Onion Salad", "Green Salad",
                        "Pickle Plate", "Garlic Naan", "Butter Roti", "Laccha Paratha"],
            "drink":   ["Lassi", "Masala Chaas", "Sweet Lassi", "Mango Lassi",
                        "Nimbu Pani", "Jaljeera"],
            "dessert": ["Gulab Jamun", "Rasmalai", "Gajar Halwa", "Kheer",
                        "Moong Dal Halwa", "Jalebi"],
        },
        "South Indian": {
            "main":    ["Masala Dosa", "Idli Sambar", "Vada Combo", "Rava Dosa",
                        "Uttapam", "Pesarattu", "Set Dosa", "Appam Stew",
                        "Chettinad Chicken", "Hyderabadi Veg Biryani"],
            "side":    ["Medu Vada", "Coconut Chutney", "Sambar Bowl",
                        "Rasam", "Pickle", "Podi"],
            "drink":   ["Filter Coffee", "Buttermilk", "Tender Coconut Water",
                        "Paneer Soda", "Rose Milk"],
            "dessert": ["Payasam", "Mysore Pak", "Kesari Bath",
                        "Rava Kesari", "Badam Halwa"],
        },
        "Chinese": {
            "main":    ["Fried Rice", "Hakka Noodles", "Manchurian Gravy",
                        "Chilli Chicken", "Sweet Corn Soup", "Schezwan Noodles",
                        "Dragon Chicken", "Kung Pao Paneer", "Crispy Honey Chilli Potato",
                        "Burnt Garlic Rice"],
            "side":    ["Spring Roll", "Momos", "Dim Sum", "Wonton Fry",
                        "Prawn Crackers", "Chilli Garlic Sauce"],
            "drink":   ["Iced Tea", "Lemon Soda", "Green Tea",
                        "Cold Coffee", "Virgin Mojito"],
            "dessert": ["Chocolate Spring Roll", "Fried Ice Cream",
                        "Date Pancake", "Honey Noodles"],
        },
        "Fast Food": {
            "main":    ["Veg Burger", "Chicken Burger", "Fries Combo",
                        "Paneer Wrap", "Chicken Wrap", "Hot Dog",
                        "Pizza Slice", "Loaded Nachos", "Veg Frankie",
                        "Double Patty Burger"],
            "side":    ["French Fries", "Coleslaw", "Onion Rings",
                        "Cheese Dip", "Peri Peri Fries", "Garlic Bread"],
            "drink":   ["Coke", "Pepsi", "Sprite", "Milkshake",
                        "Cold Coffee", "Iced Latte"],
            "dessert": ["Brownie", "Sundae", "Chocolate Lava Cake",
                        "Cookie", "Waffle"],
        },
        "Italian": {
            "main":    ["Margherita Pizza", "Pasta Alfredo", "Penne Arrabiata",
                        "Four Cheese Pizza", "Lasagna", "Risotto",
                        "Spaghetti Bolognese", "Calzone", "Bruschetta Platter",
                        "Gnocchi"],
            "side":    ["Garlic Bread", "Caesar Salad", "Soup of the Day",
                        "Cheesy Dip", "Breadstick"],
            "drink":   ["Virgin Mojito", "Iced Tea", "Lemonade",
                        "Sparkling Water", "Cappuccino"],
            "dessert": ["Tiramisu", "Panna Cotta", "Gelato",
                        "Cannoli", "Chocolate Mousse"],
        },
        "Biryani": {
            "main":    ["Chicken Biryani", "Mutton Biryani", "Veg Biryani",
                        "Egg Biryani", "Hyderabadi Dum Biryani", "Lucknowi Biryani",
                        "Paneer Biryani", "Prawn Biryani", "Keema Biryani",
                        "Ambur Biryani"],
            "side":    ["Raita", "Salan", "Mirchi Ka Salan", "Onion Raita",
                        "Boiled Egg", "Rumali Roti"],
            "drink":   ["Rooh Afza", "Lassi", "Masala Chaas",
                        "Pepsi", "Lemon Soda"],
            "dessert": ["Phirni", "Double Ka Meetha", "Shahi Tukda",
                        "Qubani Ka Meetha", "Kulfi"],
        },
        "Desserts": {
            "main":    ["Waffle Platter", "Pancake Stack", "Crepe Combo",
                        "Churros Platter", "Belgian Waffle", "Ice Cream Sundae",
                        "Thick Shake Meal", "Cookie Skillet", "Brownie Tower",
                        "Nutella Dosa"],
            "side":    ["Whipped Cream", "Extra Chocolate Sauce", "Maple Syrup",
                        "Nutella Dip", "Fruit Bowl"],
            "drink":   ["Thick Shake", "Cold Coffee", "Hot Chocolate",
                        "Oreo Shake", "Strawberry Smoothie"],
            "dessert": ["Chocolate Truffle Pastry", "Red Velvet Cake Slice",
                        "Cheesecake", "Rasmalai Cake", "Blueberry Muffin"],
        },
    }

    PRICE_MAIN = {"low": (100, 300), "mid": (200, 600), "high": (400, 1500)}
    CAT_SCALE  = {"main": 1.0, "side": 0.40, "drink": 0.25, "dessert": 0.50}
    CATEGORIES = ["main", "side", "drink", "dessert"]
    CAT_PROBS  = [0.50, 0.20, 0.15, 0.15]

    rows = []
    item_id = 1

    for _, rest in restaurants.iterrows():
        n_items = rng.integers(30, 61)
        pr      = rest["price_range"]
        cuisine = rest["cuisine_type"]
        lo, hi  = PRICE_MAIN[pr]

        cats = rng.choice(CATEGORIES, size=n_items, p=CAT_PROBS)

        for cat in cats:
            scale = CAT_SCALE[cat]
            price = round(rng.uniform(lo * scale, hi * scale), 2)
            pop   = round(rng.beta(2, 5), 4)

            pool = ITEM_NAMES[cuisine][cat]
            name = rng.choice(pool)

            if cuisine in ("Desserts",):
                veg = "veg"
            elif cat in ("drink", "dessert"):
                veg = rng.choice(["veg", "non-veg"], p=[0.85, 0.15])
            else:
                veg = rng.choice(["veg", "non-veg"], p=[0.55, 0.45])

            rows.append({
                "item_id":          item_id,
                "restaurant_id":    rest["restaurant_id"],
                "item_name":        name,
                "category":         cat,
                "veg_or_nonveg":    veg,
                "price":            price,
                "popularity_score": pop,
            })
            item_id += 1

    items = pd.DataFrame(rows)
    items.to_csv(RAW / "items.csv", index=False)
    print(f"✅ items.csv  →  {items.shape}")
    print(f"   Items per restaurant: min={items.groupby('restaurant_id').size().min()}, "
          f"max={items.groupby('restaurant_id').size().max()}, "
          f"mean={items.groupby('restaurant_id').size().mean():.1f}")
    return items


def sanity_checks(users, restaurants, items):
    """Print sanity checks for all base tables."""
    print("=" * 60)
    print("USERS")
    print(f"  Shape          : {users.shape}")
    print(f"  Budget dist    : {users['budget_segment'].value_counts(normalize=True).to_dict()}")
    print(f"  Veg dist       : {users['veg_preference'].value_counts(normalize=True).to_dict()}")
    print()
    print("RESTAURANTS")
    print(f"  Shape          : {restaurants.shape}")
    print(f"  Price range    : {restaurants['price_range'].value_counts().to_dict()}")
    print(f"  Cuisine types  : {restaurants['cuisine_type'].nunique()}")
    print(f"  Chain %        : {restaurants['is_chain'].mean():.2%}")
    print()
    print("ITEMS")
    print(f"  Shape          : {items.shape}")
    print(f"  Category dist  : {items['category'].value_counts(normalize=True).round(3).to_dict()}")
    print(f"  Veg/Non-veg    : {items['veg_or_nonveg'].value_counts().to_dict()}")
    print(f"  Price range    : ₹{items['price'].min():.0f} – ₹{items['price'].max():.0f}")
    print("=" * 60)
    print("\n✅ All base tables saved to data/raw/")


def main():
    print(f"📁 Output dir: {RAW}\n")
    users       = generate_users()
    restaurants = generate_restaurants()
    items       = generate_items(restaurants)
    sanity_checks(users, restaurants, items)


if __name__ == "__main__":
    main()
