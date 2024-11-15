
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from esda.moran import Moran
from libpysal.weights import Queen
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('D:\\biostat article\\single cell lab\\Dryad\\23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv')

# Convert to GeoDataFrame if coordinates are present
if 'x' in df.columns and 'y' in df.columns:
    geometry = [Point(xy) for xy in zip(df.x, df.y)]
    gdf = gpd.GeoDataFrame(df, geometry=geometry)

    # Visualize the data
    gdf.plot(column='Community', cmap='viridis', legend=True)
    plt.show()

    # Compute Moran's I for spatial autocorrelation
    w = Queen.from_dataframe(gdf)
    moran = Moran(gdf['MUC2'], w)  # Replace 'MUC2' with the correct column
    print(f"Moran's I: {moran.I}, p-value: {moran.p_sim}")
else:
    print("No spatial coordinates found. Please check your data.")
