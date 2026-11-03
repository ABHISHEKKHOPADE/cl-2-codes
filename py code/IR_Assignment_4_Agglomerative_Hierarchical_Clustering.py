"""Python code exported from IR_Assignment_4_Agglomerative_Hierarchical_Clustering.ipynb."""

# %% Cell 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

plt.rcParams["figure.figsize"] = (10, 5)

# %% Cell 2
# Use the Iris dataset as an appropriate numerical clustering dataset.
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target

display(df.head())
print("Dataset shape:", df.shape)

# %% Cell 3
print("Columns:")
print(df.columns.tolist())

print("\nData types:")
display(df.dtypes.to_frame("dtype"))

print("\nMissing values:")
display(df.isna().sum().to_frame("missing"))

print("\nSummary statistics:")
display(df.describe().T)

# %% Cell 4
X = df[iris.feature_names].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled_df = pd.DataFrame(X_scaled, columns=iris.feature_names)
display(X_scaled_df.head())

# %% Cell 5
Z = linkage(X_scaled, method="ward")

plt.figure(figsize=(12, 6))
dendrogram(
    Z,
    labels=[str(i) for i in range(len(X_scaled))],
    leaf_rotation=90,
    leaf_font_size=7
)
plt.title("Hierarchical Clustering Dendrogram — Ward Linkage")
plt.xlabel("Sample Index")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()

# %% Cell 6
n_clusters = 3

model = AgglomerativeClustering(
    n_clusters=n_clusters,
    linkage="ward"
)

labels = model.fit_predict(X_scaled)
df["cluster"] = labels

display(df.head(10))
print("Cluster counts:")
display(df["cluster"].value_counts().sort_index())

# %% Cell 7
# Visualize using the first two standardized features.
plt.figure(figsize=(9, 6))

for cluster_id in sorted(df["cluster"].unique()):
    mask = df["cluster"] == cluster_id
    plt.scatter(
        X_scaled[mask, 0],
        X_scaled[mask, 1],
        label=f"Cluster {cluster_id}",
        alpha=0.75
    )

plt.title("Agglomerative Hierarchical Clustering")
plt.xlabel(iris.feature_names[0] + " (standardized)")
plt.ylabel(iris.feature_names[1] + " (standardized)")
plt.legend()
plt.tight_layout()
plt.show()

# %% Cell 8
silhouette = silhouette_score(X_scaled, labels)
print(f"Silhouette Score: {silhouette:.4f}")

# %% Cell 9
cluster_summary = df.groupby("cluster")[iris.feature_names].mean()
display(cluster_summary)
