import pandas as pd
import torch
import torch.nn as nn  # Import torch.nn to define neural network modules
from torch_geometric.data import Data
from torch_geometric.nn import GINConv
from sklearn.preprocessing import LabelEncoder
from scipy.spatial import Delaunay
import numpy as np

# Load your dataset
df = pd.read_csv(r'D:\biostat article\single cell lab\Dryad\23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv')

# Get unique regions from the 'Tissue Segment' column (or any other region column if relevant)
unique_regions = df['Tissue Segment'].unique()

# Initialize a list to store results for each region
results_list = []

# Define the GIN Model class (ensure this is done before looping if not defined earlier)
class GINModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GINModel, self).__init__()
        # Define a 3-hop neighborhood model with multiple layers
        self.conv1 = GINConv(nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ))

        self.conv2 = GINConv(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ))

        self.conv3 = GINConv(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ))

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Apply the GINConv layers for 3-hop neighborhood
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        
        x = self.fc(x)
        return x

# Initialize the model (input_dim, hidden_dim, output_dim must be based on your dataset)
input_dim = 12  # Number of features
hidden_dim = 64
output_dim = len(df['Cell Type em'].unique())  # Adjust to your unique labels count
model = GINModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

# Transfer model to device (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Iterate through each unique region
for region in unique_regions:
    # Subset dataframe for the current region
    region_df = df[df['Tissue Segment'] == region]
    
    # Extract coordinates and feature columns
    cell_coordinates = region_df[['x', 'y']].values
    features = region_df[['CD31', 'CD4', 'CD8', 'CD11c', 'CD44', 'CD16', 'CD3', 'CD19', 'CD45', 'CD56', 'CD69', 'Ki67']].values

    # Apply Delaunay triangulation
    tri_full = Delaunay(cell_coordinates)
    edges = []
    for simplex in tri_full.simplices:
        for i in range(len(simplex)):
            for j in range(i + 1, len(simplex)):
                edges.append([simplex[i], simplex[j]])
    
    # Convert edges and features to PyTorch tensors
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x = torch.tensor(features, dtype=torch.float)
    
    # Encode cell type labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(region_df['Cell Type'])  # Replace 'Cell Type' with your actual label column name
    y = torch.tensor(labels, dtype=torch.long)
    
    # Create PyTorch Geometric data object
    region_data = Data(x=x, edge_index=edge_index, y=y)
    
    # Transfer data to device
    region_data = region_data.to(device)
    
    # Model prediction for the region
    model.eval()
    with torch.no_grad():
        predictions = model(region_data)
        predicted_labels = predictions.argmax(dim=1).cpu().numpy()
    
    # Ensure predicted labels are within the range
    unseen_labels = set(predicted_labels) - set(label_encoder.classes_)
    if unseen_labels:
        print(f"Warning: Predicted labels contain unseen labels: {unseen_labels}")
    
    # Replace any unseen labels with a default value (e.g., -1)
    predicted_labels = np.where(np.isin(predicted_labels, label_encoder.classes_), predicted_labels, -1)
    
    # Create a column for predicted cell type
    predicted_cell_types = np.full(predicted_labels.shape, 'Unknown', dtype=object)
    
    # Assign known cell types where possible
    mask = predicted_labels != -1
    predicted_cell_types[mask] = label_encoder.inverse_transform(predicted_labels[mask])
    
    # Add the predicted cell types to the region_df
    region_df['Predicted Cell Type'] = predicted_cell_types
    
    # Append the results to the list
    results_list.append(region_df)

# Combine results from all regions
final_results_df = pd.concat(results_list, ignore_index=True)

# Save the final results to the specified path
final_results_df.to_csv(r'D:\biostat article\single cell lab\Dryad\Per_Region_Cell_Type_Predictions.csv', index=False)

# Display the first few rows to verify
print(final_results_df.head())
