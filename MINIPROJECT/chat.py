#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

#loading in the dataset
df = pd.read_csv('MINIPROJECT/Mall_Customers.csv')
df.drop(['CustomerID', 'Gender'], axis=1, inplace=True)

# Displaying the dataset
print("Dataset:")
print(df.head())

#standardizing the data
sc = StandardScaler()
X = sc.fit_transform(df)

#PCA
pca = PCA(n_components=2)
X_2 = pca.fit_transform(X)

#K-Means Clustering
clf = KMeans(n_clusters=4,n_init=10)
labels = clf.fit_predict(X_2)
centroids = clf.cluster_centers_

# Displaying the coordinates of the centroids
print("Coordinates of the Centroids:")
for i, centroid in enumerate(centroids):
    print(f"Centroid {i+1}: {centroid}")

#cluster visualization
plt.figure(figsize=(10, 8))
plt.scatter(X_2[labels == 0, 0], X_2[labels == 0, 1], s=80, c='green', label='Cluster-1')
plt.scatter(X_2[labels == 1, 0], X_2[labels == 1, 1], s=80, c='orange', label='Cluster-2')
plt.scatter(X_2[labels == 2, 0], X_2[labels == 2, 1], s=80, c='red', label='Cluster-3')
plt.scatter(X_2[labels == 3, 0], X_2[labels == 3, 1], s=80, c='purple', label='Cluster-4')
plt.scatter(centroids[:, 0], centroids[:, 1], s=400, c='black', marker='*', label='Centroids')

plt.title('Customers Clusters')
plt.xlabel('PCA Variable-1')
plt.ylabel('PCA Variable-2')
plt.legend()
plt.show()

# Adding cluster labels to the dataset
df['Cluster'] = labels

# Displaying the dataset after segmentation
print("Dataset after Segmentation:")
print(df.head())
