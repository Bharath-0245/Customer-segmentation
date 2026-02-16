# =========================
# IMPORT LIBRARIES
# =========================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA

# Load dataset
df = pd.read_csv("Mall_Customers.csv")

# Quick check
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

X_numeric = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)

le = LabelEncoder()
gender_encoded = le.fit_transform(df['Gender']) # Male=1, Female=0

# Combine scaled numeric features with gender
X_final = np.column_stack((X_scaled, gender_encoded))

# =========================
# K-MEANS CLUSTERING
# =========================

# ----- ELBOW METHOD TO FIND OPTIMAL K -----
wcss = []  # Within-Cluster Sum of Squares
K_range = range(1, 11)  # Testing K from 1 to 10

# ----- SILHOUETTE SCORE -----
silhouette_scores = []
K_range = range(2, 11)  # Silhouette not defined for K=1

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    labels = kmeans.fit_predict(X_scaled)

    wcss.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

# Plot both
plt.figure()
plt.plot(K_range, wcss, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS (Inertia)")
plt.title("Elbow Method for Optimal K")
plt.show()

plt.figure()
plt.plot(K_range, silhouette_scores, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Analysis for Optimal K")
plt.show()

# Based on Elbow and Silhouette Plots:
optimal_k = 5
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(X_scaled)

# =========================``
# HIERARCHICAL CLUSTERING
# =========================

# ----- Generate linkage matrix -----
linked = linkage(X_scaled, method='ward')

# ----- Plot dendrogram -----
plt.figure(figsize=(10, 6))
dendrogram(linked,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=True)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Customer Index")
plt.ylabel("Distance")
plt.show()

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=cluster_labels,
    cmap='viridis',
    s=50
)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Customer Segments Visualized Using PCA")
plt.colorbar(label = "Cluster")
plt.show()

df['Cluster'] = cluster_labels
cluster_summary = df.groupby('Cluster').agg({
    'Age': 'mean',
    'Annual Income (k$)': 'mean',
    'Spending Score (1-100)': 'mean'
})

print(cluster_summary)

'''cluster_names = {
    0: "Low-income Low-spending",
    1: "Young High-spending",
    2: "Premium Customers",
    3: "High-income Low-spending",
    4: "Stable Average Customers"
}

df["Segment"] = df["Cluster"].map(cluster_names)'''

print(df)
final_silhouette = silhouette_score(X_scaled, cluster_labels)
print("Final Silhouette Score:", final_silhouette)
