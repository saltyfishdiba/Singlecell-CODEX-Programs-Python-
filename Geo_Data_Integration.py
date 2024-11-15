import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi
import random

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

# Create dummy spatial data for anatomical regions/tissue segments
regions_data = {
    'region': ['Region1', 'Region2', 'Region3'],
    'geometry': [
        Polygon([(0, 0), (0, 5000), (5000, 5000), (5000, 0)]),
        Polygon([(5000, 0), (5000, 5000), (10000, 5000), (10000, 0)]),
        Polygon([(0, 5000), (0, 10000), (5000, 10000), (5000, 5000)])
    ]
}
regions_gdf = gpd.GeoDataFrame(regions_data)

# Plot the dummy anatomical regions
fig, ax = plt.subplots(figsize=(10, 10))
regions_gdf.plot(ax=ax, color='none', edgecolor='blue')
gdf.plot(ax=ax, color='red', markersize=5)
plt.title('Anatomical Regions and Cells')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.show()

# Spatial Join: Merge the dataset with geographic information (anatomical regions)
joined_gdf = gpd.sjoin(gdf, regions_gdf, how='left', op='within')

# Display the first few rows of the joined GeoDataFrame
print(joined_gdf.head())

# Overlay Analysis: Combine the dataset with other spatial datasets (dummy data for overlay)
tissue_segments_data = {
    'segment': ['Segment1', 'Segment2'],
    'geometry': [
        Polygon([(0, 0), (0, 10000), (10000, 10000), (10000, 0)]),
        Polygon([(10000, 0), (10000, 10000), (20000, 10000), (20000, 0)])
    ]
}
tissue_segments_gdf = gpd.GeoDataFrame(tissue_segments_data)

# Perform the overlay analysis
overlay_gdf = gpd.overlay(gdf, tissue_segments_gdf, how='intersection')

# Display the first few rows of the overlay GeoDataFrame
print(overlay_gdf.head())

# Plot the result of the overlay analysis
fig, ax = plt.subplots(figsize=(10, 10))
tissue_segments_gdf.plot(ax=ax, color='none', edgecolor='green')
overlay_gdf.plot(ax=ax, color='yellow', markersize=5)
plt.title('Overlay Analysis of Tissue Segments and Cells')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.show()
