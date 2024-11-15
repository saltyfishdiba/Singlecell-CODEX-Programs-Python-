
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset (replace with the actual path to your large dataset)
file_path = r'D:\biostat article\single cell lab\Dryad\23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv'  # Adjust to your file path
data = pd.read_csv(file_path)

# Optional: Sample the dataset to reduce the density of points (adjust fraction as needed)
# Sampling 50% of the data
data_sampled = data.sample(frac=0.2, random_state=42)

# Step 1: Manually define the colors (adjust for any missing cell types)
colors_dict = {
    'NK': '#1f77b4',
    'Enterocyte': '#1f77b4',
    'MUC1+ Enterocyte': '#aec7e8',
    'TA': '#ff7f0e',
    'CD66+ Enterocyte': '#ffbb78',
    'Paneth': '#2ca02c',
    'Smooth Muscle': '#98df8a',
    'M1 Macrophage': '#d62728',
    'Goblet': '#ff9896',
    'Neuroendocrine': '#9467bd',
    'CD57+ B': '#c5b0d5',
    'Lymphatic': '#8c564b',
    'CD8+ T': '#c49c94',
    'DC': '#e377c2',
    'M2 Macrophage': '#f7b6d2',
    'B': '#7f7f7f',
    'Neutrophil': '#bcbd22',
    'Endothelial': '#dbdb8d',
    'Cycling TA': '#17becf',
    'Plasma': '#9edae5',
    'CD4+ T': '#f0e442',
    'Stroma': '#d62728',
    'Nerve': '#2ca02c',
    'ICC': '#1f77b4',
    'CD7+ Immune': '#17becf'
}
# Step 2: Create the figure and axis with zoomed-in axis limits and increased marker size
plt.figure(figsize=(24, 20))

# Plot each cell type with the custom colors
for cell_type in data_sampled['Cell Type'].unique():
    if cell_type in colors_dict:
        subset = data_sampled[data_sampled['Cell Type'] == cell_type]
        plt.scatter(subset['x'], subset['y'], label=cell_type, color=colors_dict[cell_type], s=1, alpha=0.7, edgecolor='none')  # Increase marker size to 4

# Step 3: Customize the plot appearance
plt.title('Zoomed-in Choropleth Map of Cell Types with Adjusted Spacing')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')

# Zoom in and adjust the axis limits (modify as needed to focus on specific regions)
plt.xlim(0, 9100)  # Adjust based on the region of interest
plt.ylim(0, 9100)  # Adjust based on the region of interest

# Create a legend similar to the one in your reference
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Cell Types', markerscale=6, fontsize='small', frameon=True)

# Step 4: Display the plot
plt.show()
