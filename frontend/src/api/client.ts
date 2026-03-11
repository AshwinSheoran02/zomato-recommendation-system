import axios from 'axios'
import type {
  CatalogResponse,
  DashboardResponse,
  RecommendPayload,
  RecommendResponse,
} from '../types'

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://127.0.0.1:8000'

const client = axios.create({
  baseURL: API_BASE,
  timeout: 10000,
})

export const apiBase = API_BASE

export async function fetchCatalog(restaurantId?: number): Promise<CatalogResponse> {
  const params = restaurantId ? { restaurant_id: restaurantId } : undefined
  const { data } = await client.get<CatalogResponse>('/catalog', { params })
  return data
}

export async function fetchDashboardMetrics(): Promise<DashboardResponse> {
  const { data } = await client.get<DashboardResponse>('/dashboard-metrics')
  return data
}

export async function fetchRecommendations(payload: RecommendPayload): Promise<RecommendResponse> {
  const { data } = await client.post<RecommendResponse>('/recommend', payload)
  return data
}
