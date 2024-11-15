import pandas as pd
import scanpy as sc
from matplotlib.pyplot import rc_context

# Load the CSV file into a pandas DataFrame
file_path = r'D:\biostat article\single cell lab\Dryad\23_09_CODEX_Merged0.csv'
df = pd.read_csv(file_path)

# Inspect the DataFrame columns to find a valid gene/marker name
print("Available columns in the dataset:")
print(df.columns)

# Ensure only numeric columns are used for the AnnData object
# Assuming the first column is the cell identifiers and the rest are the gene expression data
df.set_index(df.columns[0], inplace=True)  # Set the first column as index

# Fill NaN values with zeros
df.fillna(0, inplace=True)

# Select numeric columns only
numeric_cols = df.select_dtypes(include=[float, int]).columns
adata = sc.AnnData(df[numeric_cols])

# Add metadata columns back to AnnData object
metadata_cols = df.select_dtypes(exclude=[float, int]).columns
for col in metadata_cols:
    adata.obs[col] = df[col]

# Preprocess the AnnData object if necessary (e.g., normalization, log transformation)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Compute PCA with explicit n_components
sc.tl.pca(adata, svd_solver='arpack', n_comps=40)

# Compute neighbors and UMAP
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
sc.tl.umap(adata)

# Plot UMAP for a specific gene/marker
# Replace 'CD79A' with a valid gene/marker name from your dataset
valid_marker = 'CDX2'  # Example: replace with an actual marker from your dataset
with rc_context({"figure.figsize": (4, 4)}):
    sc.pl.umap(adata, color=valid_marker)
