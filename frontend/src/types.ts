export type VegPreference = 'veg' | 'non-veg' | 'mixed'
export type UserSegment = 'casual' | 'high_value' | 'cold_start' | 'warm'

export interface RestaurantOption {
  restaurant_id: number
  item_count: number
}

export interface CatalogItem {
  item_id: number
  item_name: string
  category: string
  veg_or_nonveg: string
  price: number
  popularity_score: number
}

export interface CatalogResponse {
  restaurants: RestaurantOption[]
  items: CatalogItem[]
}

export interface RecommendationItem {
  rank: number
  item_id: number
  item_name: string
  category: string
  veg_or_nonveg: string
  price: number
  popularity_score: number
  model_score?: number | null
  reason_tags: string[]
}

export interface RecommendPayload {
  restaurant_id: number
  cart_item_ids: number[]
  user_veg_preference: VegPreference
  user_segment: UserSegment
  top_k: number
  candidate_pool_size: number
}

export interface RecommendResponse {
  restaurant_id: number
  top_k: number
  latency_ms: number
  cart_summary: {
    cart_value: number
    cart_item_count: number
    has_drink: boolean
    has_dessert: boolean
    hour_of_day: number
    weekday: number
  }
  model_recommendations: RecommendationItem[]
  baseline_recommendations: RecommendationItem[]
}

export interface DashboardMetric {
  name: string
  baseline: string
  model: string
  lift: string
}

export interface DashboardResponse {
  metrics: DashboardMetric[]
  chart_urls: string[]
}
