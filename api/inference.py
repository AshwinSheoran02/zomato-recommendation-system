import hashlib
import time
from datetime import datetime

import numpy as np
import pandas as pd

from api.data_store import DataStore
from api.schemas import CartSummary, RecommendRequest, RecommendResponse, RecommendationItem


def _veg_match(user_pref: str, item_pref: str) -> int:
    if user_pref == "mixed":
        return 1
    return int(user_pref == item_pref)


def _reason_tags(
    row: pd.Series,
    cart_summary: CartSummary,
    user_pref: str,
) -> list[str]:
    tags: list[str] = []
    if row["category"] == "drink" and not cart_summary.has_drink:
        tags.append("fills_missing_drink")
    if row["category"] == "dessert" and not cart_summary.has_dessert:
        tags.append("fills_missing_dessert")
    if row["popularity_score"] >= 0.7:
        tags.append("high_popularity")
    if user_pref != "mixed" and row["veg_or_nonveg"] == user_pref:
        tags.append("matches_veg_preference")
    if row["price"] < 120:
        tags.append("budget_friendly")
    if not tags:
        tags.append("complements_cart")
    return tags


def _build_cart_summary(cart_items: pd.DataFrame, request_time: datetime) -> CartSummary:
    categories = set(cart_items["category"].tolist()) if not cart_items.empty else set()
    return CartSummary(
        cart_value=float(cart_items["price"].sum()) if not cart_items.empty else 0.0,
        cart_item_count=int(len(cart_items)),
        has_drink="drink" in categories,
        has_dessert="dessert" in categories,
        hour_of_day=request_time.hour,
        weekday=request_time.weekday(),
    )


def _candidate_pool(
    store: DataStore,
    request: RecommendRequest,
) -> pd.DataFrame:
    rest_items = store.items[store.items["restaurant_id"] == request.restaurant_id].copy()
    if rest_items.empty:
        return rest_items

    if request.cart_item_ids:
        rest_items = rest_items[~rest_items["item_id"].isin(request.cart_item_ids)]
    if rest_items.empty:
        return rest_items

    pool_size = min(request.candidate_pool_size, len(rest_items))
    weights = rest_items["popularity_score"].to_numpy(dtype=float)
    if weights.sum() == 0:
        weights = np.ones_like(weights)
    weights = weights / weights.sum()

    hash_input = f"{request.restaurant_id}:{request.cart_item_ids}:{request.top_k}"
    seed = int(hashlib.sha256(hash_input.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    chosen_idx = rng.choice(len(rest_items), size=pool_size, replace=False, p=weights)
    return rest_items.iloc[chosen_idx].copy()


def _feature_frame(
    candidates: pd.DataFrame,
    cart_summary: CartSummary,
    request: RecommendRequest,
    store: DataStore,
) -> pd.DataFrame:
    features = pd.DataFrame(
        {
            "cart_value": cart_summary.cart_value,
            "cart_item_count": cart_summary.cart_item_count,
            "has_drink": int(cart_summary.has_drink),
            "has_dessert": int(cart_summary.has_dessert),
            "hour_of_day": cart_summary.hour_of_day,
            "weekday": cart_summary.weekday,
            "candidate_price": candidates["price"].astype(float).values,
            "candidate_category": candidates["category"]
            .map(store.category_to_id)
            .fillna(-1)
            .astype(int)
            .values,
            "candidate_popularity": candidates["popularity_score"].astype(float).values,
            "matches_user_veg_pref": candidates["veg_or_nonveg"]
            .map(lambda value: _veg_match(request.user_veg_preference, value))
            .astype(int)
            .values,
        }
    )
    return features[store.feature_cols]


def _to_response_items(
    ranked_df: pd.DataFrame,
    cart_summary: CartSummary,
    user_pref: str,
    top_k: int,
    include_score: bool,
) -> list[RecommendationItem]:
    ranked_df = ranked_df.head(top_k).reset_index(drop=True)
    out: list[RecommendationItem] = []
    for idx, row in ranked_df.iterrows():
        out.append(
            RecommendationItem(
                rank=idx + 1,
                item_id=int(row["item_id"]),
                item_name=str(row["item_name"]),
                category=str(row["category"]),
                veg_or_nonveg=str(row["veg_or_nonveg"]),
                price=float(row["price"]),
                popularity_score=float(row["popularity_score"]),
                model_score=float(row["model_score"]) if include_score else None,
                reason_tags=_reason_tags(row, cart_summary, user_pref),
            )
        )
    return out


def recommend(store: DataStore, request: RecommendRequest) -> RecommendResponse:
    t0 = time.perf_counter()
    request_time = request.request_timestamp or datetime.utcnow()

    rest_items = store.items[store.items["restaurant_id"] == request.restaurant_id]
    if rest_items.empty:
        raise ValueError(f"Unknown restaurant_id={request.restaurant_id}")

    cart_items = (
        rest_items[rest_items["item_id"].isin(request.cart_item_ids)].copy()
        if request.cart_item_ids
        else rest_items.iloc[0:0].copy()
    )
    cart_summary = _build_cart_summary(cart_items, request_time)

    candidates = _candidate_pool(store, request)
    if candidates.empty:
        latency_ms = (time.perf_counter() - t0) * 1000
        return RecommendResponse(
            restaurant_id=request.restaurant_id,
            top_k=request.top_k,
            latency_ms=round(latency_ms, 3),
            cart_summary=cart_summary,
            model_recommendations=[],
            baseline_recommendations=[],
        )

    feature_df = _feature_frame(candidates, cart_summary, request, store)
    candidates["model_score"] = store.model.predict_proba(feature_df)[:, 1]
    model_ranked = candidates.sort_values("model_score", ascending=False)

    baseline_df = store.baseline_top10[
        store.baseline_top10["restaurant_id"] == request.restaurant_id
    ].copy()
    if request.cart_item_ids:
        baseline_df = baseline_df[~baseline_df["item_id"].isin(request.cart_item_ids)]
    baseline_df = baseline_df.merge(
        store.items[["item_id", "restaurant_id", "item_name", "veg_or_nonveg"]],
        on=["item_id", "restaurant_id"],
        how="left",
    )
    if len(baseline_df) < request.top_k:
        baseline_ids = set(baseline_df["item_id"].tolist())
        fallback_df = store.items[
            (store.items["restaurant_id"] == request.restaurant_id)
            & (~store.items["item_id"].isin(request.cart_item_ids))
            & (~store.items["item_id"].isin(list(baseline_ids)))
        ].copy()
        fallback_df = fallback_df.sort_values("popularity_score", ascending=False)
        if not fallback_df.empty:
            fallback_df = fallback_df.head(request.top_k - len(baseline_df))
            fallback_df["rank"] = np.arange(len(baseline_df) + 1, len(baseline_df) + len(fallback_df) + 1)
            baseline_df = pd.concat(
                [
                    baseline_df[
                        ["restaurant_id", "rank", "item_id", "category", "price", "popularity_score", "item_name", "veg_or_nonveg"]
                    ],
                    fallback_df[
                        ["restaurant_id", "rank", "item_id", "category", "price", "popularity_score", "item_name", "veg_or_nonveg"]
                    ],
                ],
                ignore_index=True,
            )
    baseline_df["model_score"] = baseline_df["popularity_score"].astype(float)
    baseline_ranked = baseline_df.sort_values(["rank", "popularity_score"], ascending=[True, False])

    latency_ms = (time.perf_counter() - t0) * 1000
    return RecommendResponse(
        restaurant_id=request.restaurant_id,
        top_k=request.top_k,
        latency_ms=round(latency_ms, 3),
        cart_summary=cart_summary,
        model_recommendations=_to_response_items(
            model_ranked, cart_summary, request.user_veg_preference, request.top_k, include_score=True
        ),
        baseline_recommendations=_to_response_items(
            baseline_ranked, cart_summary, request.user_veg_preference, request.top_k, include_score=False
        ),
    )
