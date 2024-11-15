import pandas as pd
import torch
from torch_geometric.data import Data

# Load your dataset
file_path = r'D:\biostat article\single cell lab\Dryad\23_09_CODEX_Merged0.csv'
data_df = pd.read_csv(file_path)

# Extract relevant columns, assuming you have columns named 'x', 'y', and 'cell_type'
# You can adjust these names based on your dataset structure
x_coords = data_df['x'].values
y_coords = data_df['y'].values
cell_types = data_df['Cell type'].values

# Normalize x and y to fit within the 9300 x 9300 scale
data_df['x'] = data_df['x'] / data_df['x'].max() * 9300
data_df['y'] = data_df['y'] / data_df['y'].max() * 9300

# Convert the dataset into a graph structure
# Assuming each cell is a node and edges are determined by spatial proximity
# For simplicity, we'll assume an edge exists if two nodes are within a certain distance
from sklearn.neighbors import NearestNeighbors

# Define your desired radius for connections
radius = 500  # Adjust this as needed

# Using NearestNeighbors to determine adjacency
coords = data_df[['x', 'y']].values
nbrs = NearestNeighbors(radius=radius).fit(coords)
adjacency_matrix = nbrs.radius_neighbors_graph(coords).toarray()

# Create edge_index based on adjacency matrix
edge_index = torch.tensor([[i, j] for i in range(adjacency_matrix.shape[0])
                           for j in range(adjacency_matrix.shape[1]) if adjacency_matrix[i, j]], dtype=torch.long).t().contiguous()

# Create node features using cell types or any other feature column
node_features = torch.tensor(pd.get_dummies(data_df['cell_type']).values, dtype=torch.float)

# Create a PyTorch Geometric Data object
data = Data(x=node_features, edge_index=edge_index)

# Use the data object for your GNN model
print(data)
