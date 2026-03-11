import type { RecommendationItem } from '../types'

interface Props {
  title: string
  rows: RecommendationItem[]
  showScore: boolean
}

export function RecommendationTable({ title, rows, showScore }: Props) {
  return (
    <section className="card">
      <h2>{title}</h2>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Rank</th>
              <th>Item</th>
              <th>Category</th>
              <th>Price</th>
              {showScore ? <th>Model Score</th> : <th>Popularity</th>}
              <th>Reason Tags</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((item) => (
              <tr key={`${title}-${item.item_id}-${item.rank}`}>
                <td>{item.rank}</td>
                <td>
                  <strong>{item.item_name}</strong>
                </td>
                <td>{item.category}</td>
                <td>₹{item.price.toFixed(0)}</td>
                <td>
                  {showScore
                    ? (item.model_score ?? 0).toFixed(3)
                    : item.popularity_score.toFixed(3)}
                </td>
                <td>{item.reason_tags.join(', ')}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}
