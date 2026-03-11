from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


VegPreference = Literal["veg", "non-veg", "mixed"]
UserSegment = Literal["casual", "high_value", "cold_start", "warm"]


class RecommendRequest(BaseModel):
    restaurant_id: int = Field(..., gt=0)
    cart_item_ids: List[int] = Field(default_factory=list)
    user_veg_preference: VegPreference = "mixed"
    user_segment: UserSegment = "casual"
    request_timestamp: Optional[datetime] = None
    top_k: int = Field(default=10, ge=1, le=20)
    candidate_pool_size: int = Field(default=40, ge=10, le=100)

    @field_validator("cart_item_ids")
    @classmethod
    def unique_cart_items(cls, value: List[int]) -> List[int]:
        return list(dict.fromkeys(value))


class RecommendationItem(BaseModel):
    rank: int
    item_id: int
    item_name: str
    category: str
    veg_or_nonveg: str
    price: float
    popularity_score: float
    model_score: Optional[float] = None
    reason_tags: List[str] = Field(default_factory=list)


class CartSummary(BaseModel):
    cart_value: float
    cart_item_count: int
    has_drink: bool
    has_dessert: bool
    hour_of_day: int
    weekday: int


class RecommendResponse(BaseModel):
    restaurant_id: int
    top_k: int
    latency_ms: float
    cart_summary: CartSummary
    model_recommendations: List[RecommendationItem]
    baseline_recommendations: List[RecommendationItem]


class RestaurantOption(BaseModel):
    restaurant_id: int
    item_count: int


class CatalogItem(BaseModel):
    item_id: int
    item_name: str
    category: str
    veg_or_nonveg: str
    price: float
    popularity_score: float


class CatalogResponse(BaseModel):
    restaurants: List[RestaurantOption]
    items: List[CatalogItem]


class DashboardMetric(BaseModel):
    name: str
    baseline: str
    model: str
    lift: str


class DashboardResponse(BaseModel):
    metrics: List[DashboardMetric]
    chart_urls: List[str]
