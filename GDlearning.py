import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from geovoronoi import voronoi_regions_from_coords
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GINConv
from torch_geometric.utils import from_networkx, subgraph
import networkx as nx
import random
from scipy.spatial import Voronoi

# Step 1: Load the smaller dataset
file_path = r'D:\biostat article\single cell lab\Dryad\23_09_CODEX_Merged0.csv'
data = pd.read_csv(file_path)

# Step 2: Sample the dataset (optional, reduce density if needed)
data_sampled = data.sample(frac=0.5, random_state=42)  # Adjust the fraction as needed

# Step 3: Convert x and y coordinates to Shapely Points
coords = [Point(xy) for xy in data_sampled[['x', 'y']].values]

# Step 4: Define the geographic boundary (geo_shape) as a rectangle or other shape
geo_shape = Polygon([(0, 0), (10000, 0), (10000, 10000), (0, 10000)])

# Step 5: Generate Voronoi regions using scipy.spatial.Voronoi
vor = Voronoi(data_sampled[['x', 'y']].values)

# Step 6: Visualize the Voronoi diagram with cells
plt.figure(figsize=(12, 12))

regions, vertices = voronoi_regions_from_coords(coords, geo_shape)

for region in regions:
    polygon = vertices[region]
    
    if isinstance(polygon, np.ndarray):  # Check if the polygon is iterable (NumPy array)
        plt.fill(*zip(*polygon), alpha=0.4)
    else:
        print(f"Warning: Skipping non-iterable polygon {polygon}")

# Plot the sampled points on the Voronoi diagram
plt.scatter(data_sampled['x'], data_sampled['y'], color='red', s=1)
plt.title('Voronoi Diagram of Cells (Smaller Dataset)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.show()

# Step 7: Construct the graph from the Voronoi diagram using Voronoi ridge points
G = nx.Graph()

# Add nodes with features (e.g., cell type)
for idx, row in data_sampled.iterrows():
    G.add_node(idx, x=row['x'], y=row['y'], cell_type=row['Cell Type'])

# Step 8: Add edges between neighboring Voronoi cells, but only if the points are within the bounds
num_nodes = len(data_sampled)
valid_edges = []
for ridge_points in vor.ridge_points:
    point1, point2 = ridge_points
    # Ensure both points are valid node indices
    if point1 < num_nodes and point2 < num_nodes:
        valid_edges.append((point1, point2))
        G.add_edge(point1, point2)

# Step 9: Prepare positions for graph visualization
# Create a position dictionary for nodes that are present in the sampled data
pos = {i: (data_sampled.iloc[i]['x'], data_sampled.iloc[i]['y']) for i in range(len(data_sampled))}

# Ensure all nodes in the graph have a position; if a node has no position, assign a default value
for node in G.nodes():
    if node not in pos:
        pos[node] = (0, 0)  # Assign a default position if not found in data_sampled

# Step 10: Ensure all nodes have the same attributes (x, y, cell_type)
for node, data in G.nodes(data=True):
    if 'x' not in data:
        G.nodes[node]['x'] = 0  # Assign default x value
    if 'y' not in data:
        G.nodes[node]['y'] = 0  # Assign default y value
    if 'cell_type' not in data:
        G.nodes[node]['cell_type'] = None  # Assign default cell_type value

# Step 11: Convert graph to PyTorch Geometric format
pyg_data = from_networkx(G)

# Step 12: Define the GNN model
class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = GINConv(torch.nn.Sequential(torch.nn.Linear(3, 32), torch.nn.ReLU(), torch.nn.Linear(32, 64)))
        self.conv2 = GINConv(torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 128)))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x

# Step 13: Set up data for training the GNN
# Use node features (e.g., x, y coordinates, and cell type encoded)
cell_type_mapping = {cell_type: i for i, cell_type in enumerate(data_sampled['Cell Type'].unique())}
data_sampled['cell_type_encoded'] = data_sampled['Cell Type'].map(cell_type_mapping)

# Assign node features (x, y, cell type) as input to GNN
node_features = np.stack([data_sampled['x'], data_sampled['y'], data_sampled['cell_type_encoded']], axis=1)
pyg_data.x = torch.tensor(node_features, dtype=torch.float)

# Random target for simplicity (replace with actual biological outcomes if available)
pyg_data.y = torch.tensor([random.choice([0, 1]) for _ in range(len(G))], dtype=torch.long)

# Step 14: Filter valid edge indices to only include nodes that exist in `pyg_data.x`
max_node_idx = pyg_data.x.size(0)  # Get the number of nodes in the feature matrix
valid_edges_mask = (pyg_data.edge_index < max_node_idx).all(dim=0)  # Mask for valid edges
pyg_data.edge_index = pyg_data.edge_index[:, valid_edges_mask]

# Step 15: Remap `edge_index` to valid node indices
valid_node_indices = torch.unique(pyg_data.edge_index)  # Get unique valid node indices
pyg_data.x = pyg_data.x[valid_node_indices]  # Filter node features
pyg_data.y = pyg_data.y[valid_node_indices]  # Filter the target labels to match

# Remap edge indices after filtering the valid nodes
remapped_indices = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_node_indices.tolist())}
new_edge_index = torch.tensor([[remapped_indices[i.item()] for i in edge] for edge in pyg_data.edge_index.T], dtype=torch.long).T
pyg_data.edge_index = new_edge_index

# Step 16: Train the GNN model
model = GNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
model.train()
for epoch in range(10):  # Adjust number of epochs as needed
    optimizer.zero_grad()
    out = model(pyg_data)
    loss = criterion(out, pyg_data.y)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Step 17: Visualize the embeddings
with torch.no_grad():
    model.eval()
    embeddings = model(pyg_data)

    plt.figure(figsize=(8, 8))
    plt.scatter(embeddings[:, 0].cpu(), embeddings[:, 1].cpu(), c=pyg_data.y.cpu(), cmap='coolwarm', s=10)
    plt.title('Embeddings of Cell Microenvironments (Smaller Dataset)')
    plt.xlabel('Embedding Dimension 1')
    plt.ylabel('Embedding Dimension 2')
    plt.colorbar(label='Cell Type')
    plt.show()
