import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans


customer_df = pd.read_csv('Customer.csv')

customer_df = customer_df.drop('CustomerID', axis=1)
customer_df = customer_df.rename(index=str, columns={
    'Annual Income (k$)': 'Income',
    'Spending Score (1-100)': 'SpendingScore'
})



def binary_map(x):
    return x.map({'Male': 1, "Female": 0})

variablelist =  ['Gender']
customer_df[variablelist] = customer_df[variablelist].apply(binary_map)

ssd = []
range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
for num_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=num_clusters, max_iter=20)
    kmeans.fit(customer_df)
    
    ssd.append(kmeans.inertia_)


kmeans.n_clusters = 6
kmeans.fit(customer_df)

kmean_clusters = kmeans.labels_
kmean_centroids = kmeans.cluster_centers_

customer_df['cluster_id'] = kmeans.labels_

