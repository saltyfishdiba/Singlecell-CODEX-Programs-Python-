import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.data import Data, DataLoader
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib.colors import ListedColormap
from matplotlib import cm
from PIL import Image

# GIN
class GINModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GINModel, self).__init__()
        self.conv1 = GINConv(Sequential(
            Linear(input_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU()
        ))
        self.conv2 = GINConv(Sequential(
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU()
        ))
        self.conv3 = GINConv(Sequential(
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU()
        ))
        self.fc = Sequential(
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, output_dim)
        )
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        x = self.fc(x)
        return x


data_path = "D:/biostat article/single cell lab/Dryad/23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv"
data_df = pd.read_csv(data_path)

# regions
unique_regions = data_df['unique_region'].unique()
selected_regions = unique_regions[:9]  # Select the first 9 regions for example
x_col, y_col = 'x', 'y'
cell_type_col = 'Cell Type'

# Combine all regions 
all_region_df = data_df[data_df['unique_region'].isin(selected_regions)].copy()

encoder = OneHotEncoder()
cell_type_encoded = encoder.fit_transform(all_region_df[[cell_type_col]]).toarray()

marker_cols = [col for col in all_region_df.columns if col not in [x_col, y_col, cell_type_col]]
marker_features = all_region_df[marker_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
node_features = np.hstack([cell_type_encoded, marker_features])

kmeans = KMeans(n_clusters=20, random_state=0)
all_region_df['Cluster'] = kmeans.fit_predict(node_features)

cmap = ListedColormap(sns.color_palette("tab20", n_colors=20))

#Voronoi
for region in selected_regions:
    print(f"Processing unique region: {region}")
    
    region_df = all_region_df[all_region_df['unique_region'] == region]
    
    points = region_df[[x_col, y_col]].values
    vor = Voronoi(points)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='black', line_width=0.5, line_alpha=0.8, point_size=0)
    
    for point_index, region_index in enumerate(vor.point_region):
        region_vertices = vor.regions[region_index]
        if -1 not in region_vertices:  # Ensure the region is finite
            polygon = [vor.vertices[i] for i in region_vertices]
            if point_index < len(region_df['Cluster']): 
                cluster_color = cmap(region_df['Cluster'].iloc[point_index])
                ax.fill(*zip(*polygon), color=cluster_color, alpha=0.6)
    
    ax.set_xlim(region_df['x'].min(), region_df['x'].max())
    ax.set_ylim(region_df['y'].min(), region_df['y'].max())

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=19))  # Adjust vmax to 19 (20 clusters)
    sm.set_array([])  
    cbar = fig.colorbar(sm, ax=ax, ticks=range(20))  # Adjust ticks to range(20)
    cbar.set_label("Cluster")

    plt.title(f"Filled Voronoi Diagram of Cell Clusters (Region: {region})")
    plt.show()

enrichment_matrix = pd.crosstab(all_region_df['Cluster'], all_region_df[cell_type_col], normalize='index')
plt.figure(figsize=(12, 8))
sns.heatmap(enrichment_matrix, cmap='coolwarm', annot=False, linewidths=0.5, linecolor='black')
plt.title("Cell-Type Enrichment Across Clusters (Combined for All Regions)")
plt.xlabel("Cell Type")
plt.ylabel("Cluster")
plt.show()
