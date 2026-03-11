import { apiBase } from '../api/client'

interface Props {
  chartUrls: string[]
}

export function ChartsGallery({ chartUrls }: Props) {
  return (
    <section className="card">
      <h2>Evaluation Charts</h2>
      <div className="chart-grid">
        {chartUrls.map((url) => (
          <figure key={url} className="chart-item">
            <img src={`${apiBase}${url}`} alt={url} />
            <figcaption>{url.split('/').pop()}</figcaption>
          </figure>
        ))}
      </div>
    </section>
  )
}
