import type { DashboardMetric } from '../types'

interface Props {
  metrics: DashboardMetric[]
}

export function KpiCards({ metrics }: Props) {
  return (
    <section className="card">
      <h2>Business Impact Snapshot</h2>
      <div className="kpi-grid">
        {metrics.map((metric) => (
          <article key={metric.name} className="kpi-item">
            <h3>{metric.name}</h3>
            <p>
              <span>Baseline:</span> {metric.baseline}
            </p>
            <p>
              <span>Model:</span> {metric.model}
            </p>
            <p className="lift">{metric.lift}</p>
          </article>
        ))}
      </div>
    </section>
  )
}
