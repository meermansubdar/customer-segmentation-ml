# CUSTOMER SEGMENTATION USING CLUSTERING

# Step 1: Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Step 2: Load Dataset

data = pd.read_csv("D:/SRM/SRM Project/customer-segmentation-ml-project/data/Mall_Customers.csv")

print("First 5 Rows:")
print(data.head())


# Step 3: Dataset Information

print("\nDataset Info")
print(data.info())

print("\nMissing Values")
print(data.isnull().sum())


# Step 4: Convert Gender to Numeric

data['Gender'] = data['Gender'].map({'Male':0, 'Female':1})


# Step 5: Select Features for Clustering

X = data[['Age','Annual Income (k$)','Spending Score (1-100)']]


# Step 6: Normalize Data

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Step 7: Elbow Method to Find Optimal Clusters

wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(6,4))
plt.plot(range(1,11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()


# Step 8: Apply K-Means Clustering

kmeans = KMeans(n_clusters=5, random_state=42)

clusters = kmeans.fit_predict(X_scaled)

data['Cluster'] = clusters


# Step 9: View Clustered Data

print(data.head())


# Step 10: Visualization

plt.figure(figsize=(6,5))

plt.scatter(data['Annual Income (k$)'],
            data['Spending Score (1-100)'],
            c=data['Cluster'])

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation")

plt.show()


# Step 11: PCA for Visualization

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

data['PCA1'] = X_pca[:,0]
data['PCA2'] = X_pca[:,1]


# Step 12: PCA Plot

plt.figure(figsize=(6,5))

sns.scatterplot(
    x='PCA1',
    y='PCA2',
    hue='Cluster',
    data=data,
    palette='Set1'
)

plt.title("Customer Segmentation using PCA")

plt.show()


# Step 13: Cluster Analysis

cluster_summary = data.groupby('Cluster').mean()

print("\nCluster Summary")
print(cluster_summary)


# Step 14: Save Results

data.to_csv("D:/SRM/SRM Project/customer-segmentation-ml-project/output/segmented_customers.csv", index=False)

print("\nSegmented dataset saved successfully.")