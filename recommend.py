import pickle
import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm
import math

# Load the NetworkX graph
with open('movie_graph.pkl', 'rb') as f:
    G = pickle.load(f)

hetero_data = HeteroData()

# Mappings and feature placeholders
user_map = {}
movie_map = {}
user_features = []
movie_features = []

# Process nodes
for node_id, attrs in G.nodes(data=True):
    if str(node_id).startswith("user_"):
        idx = len(user_map)
        user_map[node_id] = idx
        user_features.append([0])  # Dummy user feature
    elif str(node_id).startswith("movie_"):
        idx = len(movie_map)
        movie_map[node_id] = idx
        genre_vec = []
        # Safely handle genre values
        for val in attrs["genres"].values():
            try:
                genre_vec.append(int(float(val)))  # Convert to int, handle string or NaN
            except (ValueError, TypeError):
                genre_vec.append(0)  # Default to 0 if invalid
        movie_features.append(genre_vec)

# Convert features to tensors
hetero_data["user"].x = torch.tensor(user_features, dtype=torch.float)
hetero_data["movie"].x = torch.tensor(movie_features, dtype=torch.float)

# Prepare edge indices
user_indices = []
movie_indices = []

for u, v, _ in tqdm(G.edges(data=True)):
    if str(u).startswith("user_") and str(v).startswith("movie_"):
        user_indices.append(user_map[u])
        movie_indices.append(movie_map[v])
    elif str(v).startswith("user_") and str(u).startswith("movie_"):
        user_indices.append(user_map[v])
        movie_indices.append(movie_map[u])

# Add edges to hetero graph (both directions)
hetero_data["user", "rates", "movie"].edge_index = torch.tensor([user_indices, movie_indices], dtype=torch.long)
hetero_data["movie", "rev_rates", "user"].edge_index = torch.tensor([movie_indices, user_indices], dtype=torch.long)

# Save the graph
torch.save(hetero_data, "hetero_graph.pt")
print("✅ Heterogeneous graph saved as 'hetero_graph.pt'")
