import pandas as pd
import networkx as nx
import pickle
import matplotlib.pyplot as plt
from networkx.algorithms import bipartite

# Load data
ratings_df = pd.read_csv('ml-100k/u.data', sep='\t', names=["user_id", "movie_id", "rating", "timestamp"])
movies_df = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1', header=None, 
                        usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], names=["movie_id", "title", "unknown", "Action", "Adventure", "Animation", 
                               "Children", "Comedy", "Crime", "Drama"])

# Initialize the graph
G = nx.Graph()

# Add movie nodes with genres as attributes
for _, row in movies_df.iterrows():
    movie_node = f"movie_{row['movie_id']}"
    genres = row[3:].to_dict()  # Genre columns
    G.add_node(movie_node, title=row["title"], genres=genres)

# Add user nodes and edges (ratings)
for _, row in ratings_df.iterrows():
    user_node = f"user_{row['user_id']}"
    movie_node = f"movie_{row['movie_id']}"

    if not G.has_node(user_node):
        G.add_node(user_node, type="user")

    G.add_edge(user_node, movie_node, weight=row["rating"])

# Save the full graph
with open('movie_graph.pkl', 'wb') as f:
    pickle.dump(G, f)

print("Graph successfully built, saved as 'movie_graph.pkl'")

# --- Sample bipartite subgraph visualization ---

# Extract nodes
user_nodes = [n for n in G.nodes if str(n).startswith("user_")]
movie_nodes = [n for n in G.nodes if str(n).startswith("movie_")]

# Pick first 10 of each for simplicity
sample_users = user_nodes[:10]
sample_movies = movie_nodes[:10]

# Extract edges between them
sample_edges = [(u, m) for u in sample_users for m in sample_movies if G.has_edge(u, m)]
sample_subgraph = G.edge_subgraph(sample_edges).copy()

# Color map for nodes
color_map = ['skyblue' if n.startswith('user_') else 'lightgreen' for n in sample_subgraph.nodes]

# Layout and plot
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(sample_subgraph, seed=42)
nx.draw(sample_subgraph, pos, with_labels=True, node_color=color_map, node_size=1200, font_size=8)
nx.draw_networkx_edge_labels(sample_subgraph, pos, edge_labels={(u, v): f"{d['weight']}" for u, v, d in sample_subgraph.edges(data=True)})
plt.title("Sample Bipartite Graph (10 Users x 10 Movies)")
plt.show()