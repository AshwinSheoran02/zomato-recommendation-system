import type { CatalogItem } from '../types'

interface Props {
  items: CatalogItem[]
  cartItemIds: number[]
  onToggleItem: (itemId: number) => void
}

export function CartBuilder({ items, cartItemIds, onToggleItem }: Props) {
  const selectedSet = new Set(cartItemIds)

  return (
    <section className="card">
      <h2>Cart Builder</h2>
      <p className="muted">Pick current cart items. Recommendations exclude these selections.</p>
      <div className="item-grid">
        {items.slice(0, 24).map((item) => (
          <button
            key={item.item_id}
            className={selectedSet.has(item.item_id) ? 'chip active' : 'chip'}
            onClick={() => onToggleItem(item.item_id)}
            type="button"
          >
            <span>{item.item_name}</span>
            <small>
              {item.category} · ₹{item.price.toFixed(0)}
            </small>
          </button>
        ))}
      </div>
    </section>
  )
}
