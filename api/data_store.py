import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent


@dataclass
class DataStore:
    items: pd.DataFrame
    baseline_top10: pd.DataFrame
    feature_cols: list[str]
    model: object
    category_to_id: dict[str, int]

    @classmethod
    def load(cls) -> "DataStore":
        items = pd.read_csv(ROOT / "data" / "raw" / "items.csv")
        baseline_top10 = pd.read_csv(ROOT / "data" / "processed" / "baseline_top10.csv")

        with open(ROOT / "models" / "feature_list.json", "r", encoding="utf-8") as f:
            feature_cols = json.load(f)

        model = joblib.load(ROOT / "models" / "lightgbm_model.pkl")
        categories = sorted(items["category"].astype(str).unique().tolist())
        category_to_id = {cat: idx for idx, cat in enumerate(categories)}
        return cls(
            items=items,
            baseline_top10=baseline_top10,
            feature_cols=feature_cols,
            model=model,
            category_to_id=category_to_id,
        )
