# STEP 1: Import Libraries

import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# STEP 2: Load Dataset
data = pd.read_csv("D:/SRM/SRM Project/customer-segmentation-ml-project/data/OnlineRetail.csv", encoding='ISO-8859-1')

# STEP 3: Data Cleaning (VERY IMPORTANT)

# Remove missing values
data.dropna(inplace=True)

# Remove negative or zero quantity
data = data[data['Quantity'] > 0]

# Convert date column
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

# STEP 4: Create Total Price Column

data['TotalPrice'] = data['Quantity'] * data['UnitPrice']

# STEP 5: Calculate RFM Values
# Set reference date (latest date in dataset)
today_date = data['InvoiceDate'].max()

rfm = data.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (today_date - x.max()).days,
    'InvoiceNo': 'count',
    'TotalPrice': 'sum'
})

rfm.columns = ['Recency','Frequency','Monetary']

#STEP 6: Remove Outliers (Optional but good)
rfm = rfm[(rfm['Monetary'] > 0)]

# STEP 7: Scale the Data
scaler = StandardScaler()

rfm_scaled = scaler.fit_transform(rfm)

#STEP 8: Apply Elbow Method
wcss = []

for i in range(1,10):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(rfm_scaled)
    wcss.append(kmeans.inertia_)

import matplotlib.pyplot as plt

plt.plot(range(1,10), wcss)
plt.title("Elbow Method")
plt.show()

#STEP 9: Apply K-Means
kmeans = KMeans(n_clusters=4, random_state=42)

rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

#STEP 10: Interpret Clusters
rfm.groupby('Cluster').mean()


#STEP 11: Save Output
rfm.to_csv("D:/SRM/SRM Project/customer-segmentation-ml-project/output/rfm_segmented_customers.csv")

# Step 10: Visualization

plt.figure(figsize=(6,5))

plt.scatter(rfm['Frequency'],
            rfm['Monetary'],
            c=rfm['Cluster'])

plt.xlabel("Frequency")
plt.ylabel("Monetary")
plt.title("Customer Segmentation")

plt.show()


# Step 11: PCA for Visualization

pca = PCA(n_components=2)

X_pca = pca.fit_transform(rfm_scaled)

rfm['PCA1'] = X_pca[:,0]
rfm['PCA2'] = X_pca[:,1]


# Step 12: PCA Plot

plt.figure(figsize=(6,5))

sns.scatterplot(
    x='PCA1',
    y='PCA2',
    hue='Cluster',
    data=rfm,
    palette='Set1'
)

plt.title("Customer Segmentation using PCA")

plt.show()