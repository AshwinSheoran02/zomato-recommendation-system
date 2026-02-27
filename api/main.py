from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.data_store import DataStore
from api.inference import recommend
from api.schemas import (
    CatalogItem,
    CatalogResponse,
    DashboardMetric,
    DashboardResponse,
    RecommendRequest,
    RecommendResponse,
    RestaurantOption,
)


ROOT = Path(__file__).resolve().parent.parent

app = FastAPI(title="Zomato Hackathon Recommendation API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = DataStore.load()
app.mount("/figures", StaticFiles(directory=ROOT / "assets" / "figures"), name="figures")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/catalog", response_model=CatalogResponse)
def catalog(restaurant_id: Optional[int] = Query(default=None, gt=0)) -> CatalogResponse:
    restaurants_df = (
        store.items.groupby("restaurant_id")["item_id"]
        .count()
        .reset_index(name="item_count")
        .sort_values("restaurant_id")
    )
    restaurants = [
        RestaurantOption(restaurant_id=int(r.restaurant_id), item_count=int(r.item_count))
        for r in restaurants_df.itertuples(index=False)
    ]

    items_df = store.items.copy()
    if restaurant_id is not None:
        items_df = items_df[items_df["restaurant_id"] == restaurant_id]
    items_df = items_df.sort_values("popularity_score", ascending=False).head(200)
    items = [
        CatalogItem(
            item_id=int(row.item_id),
            item_name=str(row.item_name),
            category=str(row.category),
            veg_or_nonveg=str(row.veg_or_nonveg),
            price=float(row.price),
            popularity_score=float(row.popularity_score),
        )
        for row in items_df.itertuples(index=False)
    ]
    return CatalogResponse(restaurants=restaurants, items=items)


@app.post("/recommend", response_model=RecommendResponse)
def recommend_route(payload: RecommendRequest) -> RecommendResponse:
    try:
        return recommend(store, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/dashboard-metrics", response_model=DashboardResponse)
def dashboard_metrics() -> DashboardResponse:
    metrics = [
        DashboardMetric(name="Acceptance Rate", baseline="52.94%", model="71.75%", lift="+35.5%"),
        DashboardMetric(name="Avg Order Value", baseline="₹803.9", model="₹830.4", lift="+3.3%"),
        DashboardMetric(name="Cold-Start Acceptance", baseline="42.24%", model="61.78%", lift="+46.3%"),
        DashboardMetric(name="Latency (p95)", baseline="N/A", model="0.455 ms", lift="real-time"),
    ]
    chart_urls = [
        "/figures/precision_at_10.png",
        "/figures/acceptance_rate.png",
        "/figures/aov_comparison.png",
        "/figures/segment_acceptance.png",
        "/figures/cold_start_comparison.png",
    ]
    return DashboardResponse(metrics=metrics, chart_urls=chart_urls)
