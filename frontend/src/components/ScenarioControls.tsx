import type { UserSegment, VegPreference } from '../types'

interface Props {
  restaurantId: number
  setRestaurantId: (id: number) => void
  restaurantIds: number[]
  vegPreference: VegPreference
  setVegPreference: (value: VegPreference) => void
  userSegment: UserSegment
  setUserSegment: (value: UserSegment) => void
  topK: number
  setTopK: (value: number) => void
  onRun: () => void
  isLoading: boolean
}

const scenarios: Array<{ label: string; veg: VegPreference; segment: UserSegment }> = [
  { label: 'Cold-Start Veg User', veg: 'veg', segment: 'cold_start' },
  { label: 'High-Value Non-Veg User', veg: 'non-veg', segment: 'high_value' },
  { label: 'Warm Mixed User', veg: 'mixed', segment: 'warm' },
]

export function ScenarioControls(props: Props) {
  const {
    restaurantId,
    setRestaurantId,
    restaurantIds,
    vegPreference,
    setVegPreference,
    userSegment,
    setUserSegment,
    topK,
    setTopK,
    onRun,
    isLoading,
  } = props

  return (
    <section className="card">
      <h2>Scenario Simulator</h2>
      <div className="control-grid">
        <label>
          Restaurant
          <select value={restaurantId} onChange={(e) => setRestaurantId(Number(e.target.value))}>
            {restaurantIds.map((id) => (
              <option key={id} value={id}>
                Restaurant #{id}
              </option>
            ))}
          </select>
        </label>
        <label>
          Veg Preference
          <select
            value={vegPreference}
            onChange={(e) => setVegPreference(e.target.value as VegPreference)}
          >
            <option value="veg">veg</option>
            <option value="non-veg">non-veg</option>
            <option value="mixed">mixed</option>
          </select>
        </label>
        <label>
          User Segment
          <select value={userSegment} onChange={(e) => setUserSegment(e.target.value as UserSegment)}>
            <option value="casual">casual</option>
            <option value="high_value">high_value</option>
            <option value="cold_start">cold_start</option>
            <option value="warm">warm</option>
          </select>
        </label>
        <label>
          Top-K
          <input
            type="number"
            min={3}
            max={20}
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value))}
          />
        </label>
      </div>
      <div className="scenario-buttons">
        {scenarios.map((scenario) => (
          <button
            key={scenario.label}
            type="button"
            className="ghost"
            onClick={() => {
              setVegPreference(scenario.veg)
              setUserSegment(scenario.segment)
            }}
          >
            {scenario.label}
          </button>
        ))}
      </div>
      <button type="button" className="primary" onClick={onRun} disabled={isLoading}>
        {isLoading ? 'Scoring...' : 'Get Recommendations'}
      </button>
    </section>
  )
}
