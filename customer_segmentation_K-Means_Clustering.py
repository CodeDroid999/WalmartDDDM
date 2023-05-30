#import libraries and packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings


# Disable the FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)
# Set the value of n_init explicitly
n_init = 10  

# Load the dataset
df = pd.read_csv('CustomerLoyaltyCardData.csv')

# Select the relevant features for clustering
features = ['Annual Income (k$)', 'Spending Score (1-100)']
data = df[features]

# Implement K-Means Clustering
num_clusters = 4

# Create a K-Means model and fit it to the data
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(data)


# Print the cluster centers
print('Cluster Centers:')
print(kmeans.cluster_centers_)

# Print the cluster labels assigned to each data point
labels = kmeans.labels_
print('Cluster Labels:')
print(labels)

# Elbow Method - Calculate the within-cluster sum of squares (WCSS) for different number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)

# Plot the WCSS against the number of clusters
plt.plot(range(1, 11), wcss)
plt.title('Elbow Graph')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Assign clusters to the data points
labels = kmeans.labels_

# Plot the data points with different colors representing different clusters
plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], c=labels)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('K-Means Clustering')
plt.show()
