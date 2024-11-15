import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d

# Load the CSV file using raw string to avoid invalid escape sequences
file_path = r'D:\biostat article\single cell lab\Dryad\23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv'
data = pd.read_csv(file_path)

# Check for missing values
missing_values = data.isnull().sum()
print(missing_values[missing_values > 0])

# Create a GeoDataFrame
geometry = [Point(xy) for xy in zip(data['x'], data['y'])]
gdf = gpd.GeoDataFrame(data, geometry=geometry)

# Display the first few rows of the GeoDataFrame
print(gdf.head())

# Choropleth Maps: Plot cell types
fig, ax = plt.subplots(figsize=(10, 10))
gdf.plot(column='Cell Type', ax=ax, legend=True, cmap='Set1', markersize=5, alpha=0.6, edgecolor='k')
plt.title('Choropleth Map of Cell Types')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.show()

# Voronoi Diagrams: Generate diagrams to partition the space
points = np.column_stack([gdf['x'], gdf['y']])
vor = Voronoi(points)

# Create Voronoi regions as polygons
regions = []
for region in vor.regions:
    if not -1 in region and len(region) > 0:
        regions.append(Polygon([vor.vertices[i] for i in region]))

# Filter out regions that are not polygons
regions = [region for region in regions if region.is_valid and region.area > 0]

# Create a GeoDataFrame for Voronoi polygons
vor_gdf = gpd.GeoDataFrame(geometry=regions)

# Plot Voronoi Diagram
fig, ax = plt.subplots(figsize=(10, 10))
vor_gdf.plot(ax=ax, color='white', edgecolor='black')
gdf.plot(ax=ax, color='red', markersize=5)
plt.title('Voronoi Diagram of Cells')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.show()