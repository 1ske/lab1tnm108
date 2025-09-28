import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# Load the customer data
customer_data = pd.read_csv('shopping_data.csv')

# Explore the dataset
print("Dataset shape:", customer_data.shape)
print("\nDataset head:")
print(customer_data.head())

# Filter the data to retain only Annual Income and Spending Score columns
# Remove CustomerID (column 0), Genre (column 1), and Age (column 2)
# Keep Annual Income (column 3) and Spending Score (column 4)
data = customer_data.iloc[:, 3:5].values

print(f"\nFiltered data shape: {data.shape}")
print("First 5 rows of filtered data:")
print(data[:5])

# Create dendrogram to determine the number of clusters
plt.figure(figsize=(15, 10))
linkage_matrix = linkage(data, method='ward')
dendrogram = dendrogram(linkage_matrix, truncate_mode='level', p=5)
plt.title('Dendrogram for Shopping Data - Analysis of Optimal Clusters')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')


# Perform hierarchical clustering with optimal number of clusters
n_clusters = 5
hierarchical_cluster = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
cluster_labels = hierarchical_cluster.fit_predict(data)

# Plot the clusters
plt.figure(figsize=(12, 8))
colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
for i in range(n_clusters):
    plt.scatter(data[cluster_labels == i, 0], data[cluster_labels == i, 1], 
                c=colors[i], label=f'Cluster {i+1}', alpha=0.7)

plt.title('Hierarchical Clustering of Shopping Data')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Print cluster information
print(f"\nNumber of clusters: {n_clusters}")
for i in range(n_clusters):
    cluster_size = np.sum(cluster_labels == i)
    print(f"Cluster {i+1}: {cluster_size} customers")
    
    # Calculate cluster statistics
    cluster_data = data[cluster_labels == i]
    avg_income = np.mean(cluster_data[:, 0])
    avg_spending = np.mean(cluster_data[:, 1])
    print(f"  Average Annual Income: {avg_income:.2f}k$")
    print(f"  Average Spending Score: {avg_spending:.2f}")
    print()