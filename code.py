import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset from CSV
df = pd.read_csv('MINIPROJECT/Mall_Customers.numbers')

# Convert column names to lowercase
df.columns = df.columns.str.lower()

# Preprocess the data
label_encoder = LabelEncoder()

# Check if the columns exist in the dataset before encoding
columns_to_encode = ['Age','Annual Income','Spending Score']
for col in columns_to_encode:
    if col in df.columns:
        df[col] = label_encoder.fit_transform(df[col])
    else:
        print(f"Column '{col}' not found in the dataset.")

# Filter out the columns to be one-hot encoded
columns_to_one_hot_encode = [col for col in columns_to_encode if col in df.columns]

# Perform one-hot encoding for categorical columns
if len(columns_to_one_hot_encode) > 0:
    df_encoded = pd.get_dummies(df, columns=columns_to_one_hot_encode)
else:
    df_encoded = df

# Scale the numerical attributes
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_encoded.iloc[:, 1:])  # Exclude the 'ID' column

# Perform K-means clustering
# Specify the desired number of clusters
# Generate a random number of clusters within a desired range
num_clusters = random.randint(2, 6)

# Apply K-means clustering
kmeans = KMeans(n_clusters=4)
kmeans.fit(scaled_data)

# Add the cluster labels to the original dataframe
df['Cluster'] = kmeans.labels_

# Print the cluster centers
print("Cluster Centers:")
print(kmeans.cluster_centers_)

# Print the count of customers in each cluster
print("\nCluster Counts:")
print(df['Cluster'].value_counts())


# Visualize the clusters
plt.scatter(df['Age'], df['Spending Score'], c=df['Cluster'], cmap='viridis')
plt.xlabel('Age')
plt.ylabel('Spending Score')
plt.title('Customer Segmentation')
plt.show()

