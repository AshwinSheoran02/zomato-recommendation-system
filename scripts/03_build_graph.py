"""
🕸️ Step 03 — Build Knowledge Graph + GraphRAG Layer
=====================================================
  1. Load raw CSV tables (users, restaurants, items, orders, order_items)
  2. Construct a heterogeneous NetworkX graph with typed nodes & edges:
       • User, Restaurant, Item, Category nodes
       • ORDERED_FROM, ORDERED_ITEM, SERVES, HAS_CATEGORY, IS_COMPLEMENTARY_TO edges
  3. Compute item co-purchase complementarity from order data
  4. Provide GraphRAG retrieval via Personalized PageRank
  5. Save graph → models/knowledge_graph.gpickle

Usage:
    python scripts/03_build_graph.py
"""

import numpy as np
import pandas as pd
import pathlib
import pickle
import warnings
import time
from collections import defaultdict, Counter
from itertools import combinations

import networkx as nx

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW  = ROOT / "data" / "raw"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

GRAPH_PATH = MODELS / "knowledge_graph.gpickle"


# ══════════════════════════════════════════════════════════════════════
#  1️⃣  LOAD RAW DATA
# ══════════════════════════════════════════════════════════════════════

def load_raw_data():
    """Load all raw CSV tables."""
    users       = pd.read_csv(RAW / "users.csv")
    restaurants = pd.read_csv(RAW / "restaurants.csv")
    items       = pd.read_csv(RAW / "items.csv")
    orders      = pd.read_csv(RAW / "orders.csv", parse_dates=["order_timestamp"])
    order_items = pd.read_csv(RAW / "order_items.csv")

    print(f"✅ Loaded raw tables:")
    print(f"   users={users.shape}  restaurants={restaurants.shape}  items={items.shape}")
    print(f"   orders={orders.shape}  order_items={order_items.shape}")
    return users, restaurants, items, orders, order_items


# ══════════════════════════════════════════════════════════════════════
#  2️⃣  BUILD HETEROGENEOUS KNOWLEDGE GRAPH
# ══════════════════════════════════════════════════════════════════════

def build_knowledge_graph(users, restaurants, items, orders, order_items):
    """
    Construct a heterogeneous graph with typed nodes and weighted edges.

    Node types: User, Restaurant, Item, Category
    Edge types: ORDERED_FROM, ORDERED_ITEM, SERVES, HAS_CATEGORY, IS_COMPLEMENTARY_TO
    """
    G = nx.Graph()  # undirected for PageRank compatibility

    t0 = time.time()

    # ── Category nodes ────────────────────────────────────────────
    categories = items["category"].unique()
    for cat in categories:
        G.add_node(f"cat:{cat}", node_type="Category", name=cat)
    print(f"   📦 Added {len(categories)} Category nodes")

    # ── Restaurant nodes ──────────────────────────────────────────
    for _, r in restaurants.iterrows():
        G.add_node(
            f"rest:{r['restaurant_id']}",
            node_type="Restaurant",
            city=r["city"],
            cuisine_type=r["cuisine_type"],
            price_range=r["price_range"],
            rating=r["rating"],
            is_chain=r["is_chain"],
        )
    print(f"   🍽️  Added {len(restaurants)} Restaurant nodes")

    # ── Item nodes + SERVES & HAS_CATEGORY edges ──────────────────
    for _, it in items.iterrows():
        iid = f"item:{it['item_id']}"
        G.add_node(
            iid,
            node_type="Item",
            category=it["category"],
            veg_or_nonveg=it["veg_or_nonveg"],
            price=it["price"],
            popularity_score=it["popularity_score"],
            item_name=it["item_name"],
        )
        # SERVES edge: Restaurant → Item
        G.add_edge(f"rest:{it['restaurant_id']}", iid,
                   edge_type="SERVES", weight=1.0)
        # HAS_CATEGORY edge: Item → Category
        G.add_edge(iid, f"cat:{it['category']}",
                   edge_type="HAS_CATEGORY", weight=1.0)
    print(f"   🍕 Added {len(items)} Item nodes + SERVES/HAS_CATEGORY edges")

    # ── User nodes ────────────────────────────────────────────────
    # Sample users to keep graph manageable (top 5000 by order activity)
    user_order_counts = orders.groupby("user_id").size().nlargest(5000)
    active_users = set(user_order_counts.index)
    active_user_df = users[users["user_id"].isin(active_users)]

    for _, u in active_user_df.iterrows():
        G.add_node(
            f"user:{u['user_id']}",
            node_type="User",
            city=u["city"],
            budget_segment=u["budget_segment"],
            veg_preference=u["veg_preference"],
            avg_order_value=u["avg_order_value"],
            order_frequency=u["order_frequency"],
        )
    print(f"   👤 Added {len(active_user_df)} User nodes (top active users)")

    # ── ORDERED_FROM edges (User → Restaurant) ───────────────────
    user_rest_orders = (
        orders[orders["user_id"].isin(active_users)]
        .groupby(["user_id", "restaurant_id"])
        .size()
        .reset_index(name="order_count")
    )
    for _, row in user_rest_orders.iterrows():
        u_node = f"user:{row['user_id']}"
        r_node = f"rest:{row['restaurant_id']}"
        if G.has_node(u_node) and G.has_node(r_node):
            G.add_edge(u_node, r_node,
                       edge_type="ORDERED_FROM",
                       weight=float(row["order_count"]))
    print(f"   🔗 Added {len(user_rest_orders)} ORDERED_FROM edges")

    # ── ORDERED_ITEM edges (User → Item) ─────────────────────────
    orders_active = orders[orders["user_id"].isin(active_users)][["order_id", "user_id"]]
    oi_with_user = order_items.merge(orders_active, on="order_id")
    user_item_counts = (
        oi_with_user.groupby(["user_id", "item_id"])
        .size()
        .reset_index(name="purchase_count")
    )
    added_oi = 0
    for _, row in user_item_counts.iterrows():
        u_node = f"user:{row['user_id']}"
        i_node = f"item:{row['item_id']}"
        if G.has_node(u_node) and G.has_node(i_node):
            G.add_edge(u_node, i_node,
                       edge_type="ORDERED_ITEM",
                       weight=float(row["purchase_count"]))
            added_oi += 1
    print(f"   🛒 Added {added_oi:,} ORDERED_ITEM edges")

    elapsed = time.time() - t0
    print(f"\n   ⏱️  Base graph built in {elapsed:.1f}s")
    return G


# ══════════════════════════════════════════════════════════════════════
#  3️⃣  COMPUTE COMPLEMENTARITY EDGES
# ══════════════════════════════════════════════════════════════════════

def add_complementarity_edges(G, order_items, min_copurchase=3, max_edges=50000):
    """
    Compute IS_COMPLEMENTARY_TO edges from co-purchase patterns.

    Two items that appear together in the same order frequently are
    considered complementary (e.g., Biryani + Raita, Burger + Fries).
    """
    t0 = time.time()
    print("\n🔄 Computing item co-purchase complementarity...")

    # Group items by order
    order_groups = order_items.groupby("order_id")["item_id"].apply(list)

    # Count co-occurrences
    copurchase = Counter()
    item_freq  = Counter()

    for items_in_order in order_groups:
        unique_items = list(set(items_in_order))
        for iid in unique_items:
            item_freq[iid] += 1
        if len(unique_items) >= 2:
            for a, b in combinations(sorted(unique_items), 2):
                copurchase[(a, b)] += 1

    # Filter by minimum co-purchase count and compute PMI-like score
    comp_edges = []
    total_orders = len(order_groups)
    for (a, b), count in copurchase.items():
        if count >= min_copurchase:
            # Pointwise Mutual Information (PMI)
            p_ab = count / total_orders
            p_a  = item_freq[a] / total_orders
            p_b  = item_freq[b] / total_orders
            pmi  = np.log2(p_ab / (p_a * p_b + 1e-10))
            if pmi > 0:
                comp_edges.append((a, b, count, pmi))

    # Sort by PMI and take top edges
    comp_edges.sort(key=lambda x: x[3], reverse=True)
    comp_edges = comp_edges[:max_edges]

    added = 0
    for a, b, count, pmi in comp_edges:
        a_node = f"item:{a}"
        b_node = f"item:{b}"
        if G.has_node(a_node) and G.has_node(b_node):
            G.add_edge(a_node, b_node,
                       edge_type="IS_COMPLEMENTARY_TO",
                       weight=round(pmi, 4),
                       copurchase_count=count)
            added += 1

    elapsed = time.time() - t0
    print(f"   ✅ Added {added:,} IS_COMPLEMENTARY_TO edges  (min co-purchase={min_copurchase})")
    print(f"   ⏱️  Complementarity computed in {elapsed:.1f}s")
    return G


# ══════════════════════════════════════════════════════════════════════
#  4️⃣  GRAPHRAG RETRIEVAL (Personalized PageRank)
# ══════════════════════════════════════════════════════════════════════

def graph_rag_retrieve(G, cart_item_ids, restaurant_id, top_k=20, alpha=0.85):
    """
    Retrieve contextually relevant item candidates via Personalized PageRank.

    Seeds the PageRank teleport vector from the current cart items,
    then returns the top-K items from the same restaurant that are
    NOT already in the cart.

    Args:
        G: NetworkX knowledge graph
        cart_item_ids: list of item IDs currently in the cart
        restaurant_id: restaurant ID to constrain candidates
        top_k: number of candidates to return
        alpha: PageRank damping factor

    Returns:
        List of (item_id, graph_score) tuples, sorted descending by score.
    """
    # Build personalization dict: seed from cart items
    personalization = {}
    for iid in cart_item_ids:
        node = f"item:{iid}"
        if G.has_node(node):
            personalization[node] = 1.0

    if not personalization:
        return []

    # Run Personalized PageRank
    try:
        pr_scores = nx.pagerank(G, alpha=alpha, personalization=personalization,
                                max_iter=100, tol=1e-6)
    except nx.PowerIterationFailedConvergence:
        pr_scores = nx.pagerank(G, alpha=alpha, personalization=personalization,
                                max_iter=300, tol=1e-4)

    # Filter to items from the same restaurant, not in cart
    cart_nodes = {f"item:{iid}" for iid in cart_item_ids}
    rest_node  = f"rest:{restaurant_id}"

    candidates = []
    for node, score in pr_scores.items():
        if not node.startswith("item:"):
            continue
        if node in cart_nodes:
            continue
        # Check if item is served by this restaurant
        if G.has_edge(rest_node, node):
            item_id = int(node.split(":")[1])
            candidates.append((item_id, score))

    # Sort by score descending and take top-K
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:top_k]


# ══════════════════════════════════════════════════════════════════════
#  5️⃣  GRAPH SUMMARY & SAVE
# ══════════════════════════════════════════════════════════════════════

def print_graph_summary(G):
    """Print a detailed summary of the knowledge graph."""
    print("\n" + "=" * 65)
    print("  🕸️  KNOWLEDGE GRAPH SUMMARY")
    print("=" * 65)

    # Node type counts
    node_types = defaultdict(int)
    for _, data in G.nodes(data=True):
        node_types[data.get("node_type", "Unknown")] += 1

    print(f"\n  📊 Total Nodes: {G.number_of_nodes():,}")
    for ntype, count in sorted(node_types.items()):
        print(f"     {ntype:20s}: {count:>8,}")

    # Edge type counts
    edge_types = defaultdict(int)
    for _, _, data in G.edges(data=True):
        edge_types[data.get("edge_type", "Unknown")] += 1

    print(f"\n  📊 Total Edges: {G.number_of_edges():,}")
    for etype, count in sorted(edge_types.items()):
        print(f"     {etype:25s}: {count:>8,}")

    # Graph density
    density = nx.density(G)
    print(f"\n  📊 Graph Density: {density:.6f}")

    # Connected components
    n_components = nx.number_connected_components(G)
    largest_cc   = max(nx.connected_components(G), key=len)
    print(f"  📊 Connected Components: {n_components}")
    print(f"  📊 Largest Component: {len(largest_cc):,} nodes "
          f"({len(largest_cc)/G.number_of_nodes():.1%} of graph)")

    print("=" * 65)


def demo_graph_rag(G):
    """Run a quick demo of GraphRAG retrieval."""
    print("\n🔍 GraphRAG Demo — Personalized PageRank Retrieval")
    print("─" * 55)

    # Pick a sample restaurant and some of its items
    rest_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "Restaurant"]
    if not rest_nodes:
        print("   ⚠️  No restaurant nodes found, skipping demo.")
        return

    sample_rest = rest_nodes[0]
    rest_id = int(sample_rest.split(":")[1])

    # Get some items from this restaurant
    rest_items = [
        int(n.split(":")[1])
        for n in G.neighbors(sample_rest)
        if n.startswith("item:") and G[sample_rest][n].get("edge_type") == "SERVES"
    ]

    if len(rest_items) < 2:
        print("   ⚠️  Not enough items for demo, skipping.")
        return

    # Simulate a cart with first 2 items
    cart = rest_items[:2]
    cart_names = [G.nodes[f"item:{iid}"].get("item_name", "?") for iid in cart]

    print(f"   Restaurant ID : {rest_id}")
    print(f"   Cart items    : {cart} → {cart_names}")
    print(f"   Retrieving top-10 candidates via PersonalizedPageRank...")

    results = graph_rag_retrieve(G, cart, rest_id, top_k=10)

    if results:
        print(f"\n   {'Rank':<6} {'Item ID':<10} {'Item Name':<30} {'Graph Score':<12}")
        print(f"   {'─'*6} {'─'*10} {'─'*30} {'─'*12}")
        for rank, (iid, score) in enumerate(results, 1):
            name = G.nodes.get(f"item:{iid}", {}).get("item_name", "?")
            print(f"   {rank:<6} {iid:<10} {name:<30} {score:.8f}")
    else:
        print("   ⚠️  No candidates retrieved (graph may be disconnected).")

    print()


def save_graph(G):
    """Save the knowledge graph to disk."""
    with open(GRAPH_PATH, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = GRAPH_PATH.stat().st_size / (1024 * 1024)
    print(f"💾 Knowledge graph saved → {GRAPH_PATH}  ({size_mb:.1f} MB)")


# ══════════════════════════════════════════════════════════════════════
#  🏁  MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  🕸️  KNOWLEDGE GRAPH + GRAPHRAG BUILDER")
    print("=" * 65)
    total_start = time.time()

    # 1. Load data
    users, restaurants, items, orders, order_items = load_raw_data()

    # 2. Build base graph
    print("\n🔨 Building heterogeneous knowledge graph...")
    G = build_knowledge_graph(users, restaurants, items, orders, order_items)

    # 3. Add complementarity edges
    G = add_complementarity_edges(G, order_items, min_copurchase=3)

    # 4. Print summary
    print_graph_summary(G)

    # 5. Demo GraphRAG retrieval
    demo_graph_rag(G)

    # 6. Save
    save_graph(G)

    total_elapsed = time.time() - total_start
    print(f"\n🏁 Knowledge Graph pipeline complete — Total time: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
