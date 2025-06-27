import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero
from sklearn.metrics import classification_report
import numpy as np
import random
import warnings
from sklearn.exceptions import UndefinedMetricWarning

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = torch.load(r'C:\Users\shash\mini-project\hetero_graph.pt', weights_only=False).to(device)

# Define the GNN Encoder
class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, dropout_rate=0.3):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

# Define the full model
class GNNModel(nn.Module):
    def __init__(self, hidden_channels, embed_dim):
        super().__init__()
        self.user_embedding = nn.Embedding(data['user'].num_nodes, embed_dim)
        self.movie_embedding = nn.Embedding(data['movie'].num_nodes, embed_dim)

        self.encoder = GNNEncoder(hidden_channels)
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')

        self.project = nn.Linear(embed_dim + hidden_channels, hidden_channels)
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, 5)
        )

    def forward(self, x_dict, edge_index_dict, user_movie_pairs):
        h_dict = self.encoder(x_dict, edge_index_dict)

        user_id = user_movie_pairs[0]
        movie_id = user_movie_pairs[1]

        user_emb = torch.cat([
            h_dict['user'][user_id],
            self.user_embedding(user_id)
        ], dim=1)

        movie_emb = torch.cat([
            h_dict['movie'][movie_id],
            self.movie_embedding(movie_id)
        ], dim=1)

        user_proj = self.project(user_emb)
        movie_proj = self.project(movie_emb)

        edge_emb = torch.cat([user_proj, movie_proj], dim=1)
        return self.classifier(edge_emb)

# Edge label setup
raw_edge_attr = data['user', 'rates', 'movie'].edge_attr
valid_mask = raw_edge_attr > 0
edge_label_index = data['user', 'rates', 'movie'].edge_index[:, valid_mask]
edge_label = raw_edge_attr[valid_mask].long() - 1

# Train/Val/Test split
num_edges = edge_label.size(0)
perm = torch.randperm(num_edges)
train_idx = perm[:int(0.7 * num_edges)]
val_idx = perm[int(0.7 * num_edges):int(0.85 * num_edges)]
test_idx = perm[int(0.85 * num_edges):]

train_edge_index = edge_label_index[:, train_idx]
train_edge_label = edge_label[train_idx]
val_edge_index = edge_label_index[:, val_idx]
val_edge_label = edge_label[val_idx]
test_edge_index = edge_label_index[:, test_idx]
test_edge_label = edge_label[test_idx]

# Model and training setup
model = GNNModel(hidden_channels=128, embed_dim=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()

# Training loop
print("\n🚀 Training started...")
for epoch in range(1, 101):
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict, train_edge_index)
    loss = criterion(out, train_edge_label)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0 or epoch == 1:
        model.eval()
        with torch.no_grad():
            val_out = model(data.x_dict, data.edge_index_dict, val_edge_index)
            val_loss = criterion(val_out, val_edge_label).item()
            val_pred = val_out.argmax(dim=1)
            val_acc = (val_pred == val_edge_label).float().mean().item()
            print(f"Epoch {epoch}/100, Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

# Final test evaluation
print("\n🧪 Final Evaluation on Test Data:")
model.eval()
with torch.no_grad():
    test_out = model(data.x_dict, data.edge_index_dict, test_edge_index)
    test_pred = test_out.argmax(dim=1)
    test_acc = (test_pred == test_edge_label).float().mean().item()

print(f"Test Accuracy: {test_acc:.4f}")
print(classification_report(test_edge_label.cpu(), test_pred.cpu(), digits=4, zero_division=1))
