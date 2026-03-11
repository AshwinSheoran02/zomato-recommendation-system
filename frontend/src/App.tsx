import { useEffect, useMemo, useState } from 'react'
import { motion } from 'framer-motion'
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { Flame, Gauge, Sparkles } from 'lucide-react'

import { fetchCatalog, fetchDashboardMetrics, fetchRecommendations } from './api/client'
import { CartBuilder } from './components/CartBuilder'
import { ChartsGallery } from './components/ChartsGallery'
import { KpiCards } from './components/KpiCards'
import { RecommendationTable } from './components/RecommendationTable'
import { ScenarioControls } from './components/ScenarioControls'
import type {
  CatalogItem,
  DashboardResponse,
  RecommendResponse,
  UserSegment,
  VegPreference,
} from './types'

function App() {
  const [restaurantIds, setRestaurantIds] = useState<number[]>([])
  const [restaurantId, setRestaurantId] = useState<number>(1)
  const [catalogItems, setCatalogItems] = useState<CatalogItem[]>([])
  const [dashboard, setDashboard] = useState<DashboardResponse | null>(null)
  const [cartItemIds, setCartItemIds] = useState<number[]>([])
  const [vegPreference, setVegPreference] = useState<VegPreference>('mixed')
  const [userSegment, setUserSegment] = useState<UserSegment>('casual')
  const [topK, setTopK] = useState<number>(10)
  const [result, setResult] = useState<RecommendResponse | null>(null)
  const [isLoading, setIsLoading] = useState<boolean>(false)
  const [error, setError] = useState<string>('')

  useEffect(() => {
    const boot = async () => {
      try {
        const [catalog, metrics] = await Promise.all([
          fetchCatalog(),
          fetchDashboardMetrics(),
        ])
        const ids = catalog.restaurants.map((r) => r.restaurant_id)
        setRestaurantIds(ids)
        const defaultRestaurant = ids[0] ?? 1
        setRestaurantId(defaultRestaurant)
        const initialItems = await fetchCatalog(defaultRestaurant)
        setCatalogItems(initialItems.items)
        setDashboard(metrics)
      } catch (err) {
        setError('API not reachable. Start FastAPI first.')
      }
    }
    void boot()
  }, [])

  useEffect(() => {
    const loadRestaurantItems = async () => {
      if (!restaurantId) {
        return
      }
      try {
        const next = await fetchCatalog(restaurantId)
        setCatalogItems(next.items)
        setCartItemIds([])
      } catch (err) {
        setError('Unable to load menu for selected restaurant.')
      }
    }
    void loadRestaurantItems()
  }, [restaurantId])

  const chartData = useMemo(() => {
    if (!result) {
      return []
    }
    return result.model_recommendations.slice(0, 8).map((rec) => ({
      name: `#${rec.rank}`,
      score: rec.model_score ?? 0,
    }))
  }, [result])

  const toggleCartItem = (itemId: number) => {
    setCartItemIds((prev) =>
      prev.includes(itemId) ? prev.filter((id) => id !== itemId) : [...prev, itemId],
    )
  }

  const runInference = async () => {
    setIsLoading(true)
    setError('')
    try {
      const response = await fetchRecommendations({
        restaurant_id: restaurantId,
        cart_item_ids: cartItemIds,
        user_veg_preference: vegPreference,
        user_segment: userSegment,
        top_k: topK,
        candidate_pool_size: 40,
      })
      setResult(response)
    } catch (err) {
      setError('Recommendation request failed. Check backend logs.')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <main className="app-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">Zomato Hackathon Demo</p>
          <h1>Intelligent Add-On Recommendation Experience</h1>
          <p className="muted">
            Real model scoring, baseline comparison, scenario simulation, and business-impact visuals
            in one demo-ready interface.
          </p>
        </div>
        <div className="hero-stats">
          <span>
            <Flame size={16} /> Model + Baseline
          </span>
          <span>
            <Sparkles size={16} /> Explainability Tags
          </span>
          <span>
            <Gauge size={16} /> Live Latency
          </span>
        </div>
      </header>

      {error ? <p className="error-banner">{error}</p> : null}

      {dashboard ? <KpiCards metrics={dashboard.metrics} /> : null}

      <ScenarioControls
        restaurantId={restaurantId}
        setRestaurantId={setRestaurantId}
        restaurantIds={restaurantIds}
        vegPreference={vegPreference}
        setVegPreference={setVegPreference}
        userSegment={userSegment}
        setUserSegment={setUserSegment}
        topK={topK}
        setTopK={setTopK}
        onRun={runInference}
        isLoading={isLoading}
      />

      <CartBuilder items={catalogItems} cartItemIds={cartItemIds} onToggleItem={toggleCartItem} />

      {result ? (
        <motion.section
          className="result-grid"
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35 }}
        >
          <article className="card">
            <h2>Real-Time Request Diagnostics</h2>
            <p className="muted">
              Latency: <strong>{result.latency_ms.toFixed(3)} ms</strong> · Cart value: ₹
              {result.cart_summary.cart_value.toFixed(0)} · Items: {result.cart_summary.cart_item_count}
            </p>
            <div className="mini-chart">
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis domain={[0, 1]} />
                  <Tooltip />
                  <Bar dataKey="score" fill="#ef4f5f" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </article>
          <RecommendationTable
            title="Model Recommendations"
            rows={result.model_recommendations}
            showScore
          />
          <RecommendationTable
            title="Baseline Recommendations"
            rows={result.baseline_recommendations}
            showScore={false}
          />
        </motion.section>
      ) : null}

      {dashboard ? <ChartsGallery chartUrls={dashboard.chart_urls} /> : null}
    </main>
  )
}

export default App
