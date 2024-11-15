import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Load the CSV file using raw string to avoid invalid escape sequences
file_path = r'D:\biostat article\single cell lab\Dryad\23_09_CODEX_Merged0.csv'
data = pd.read_csv(file_path)

# Check for missing values
missing_values = data.isnull().sum()
print(missing_values[missing_values > 0])

# Create a GeoDataFrame
geometry = [Point(xy) for xy in zip(data['x'], data['y'])]
gdf = gpd.GeoDataFrame(data, geometry=geometry)

# Display the first few rows of the GeoDataFrame
print(gdf.head())

# Plot the spatial distribution of cells
fig, ax = plt.subplots(figsize=(10, 10))
gdf.plot(ax=ax, markersize=5, alpha=0.6, edgecolor='k')
plt.title('Spatial Distribution of Cells')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.show()

# Spatial clustering using DBSCAN
coords = gdf[['x', 'y']].values
db = DBSCAN(eps=50, min_samples=10).fit(coords)
labels = db.labels_

# Add cluster labels to GeoDataFrame
gdf['cluster'] = labels

# Plot the clusters
fig, ax = plt.subplots(figsize=(10, 10))
gdf.plot(column='cluster', ax=ax, legend=True, cmap='tab20', markersize=5, alpha=0.6, edgecolor='k')
plt.title('DBSCAN Clustering of Cells')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.show()

# Nearest Neighbor Analysis
neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(coords)
distances, indices = nbrs.kneighbors(coords)

# Plot the nearest neighbor distances
plt.figure(figsize=(10, 6))
plt.hist(distances[:, 1], bins=50, edgecolor='black')
plt.title('Histogram of Nearest Neighbor Distances')
plt.xlabel('Distance')
plt.ylabel('Frequency')
plt.show()

# Mean nearest neighbor distance
mean_nnd = np.mean(distances[:, 1])
print(f'Mean Nearest Neighbor Distance: {mean_nnd}')
