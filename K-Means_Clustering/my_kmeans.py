# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 17:30:19 2017

@author: Mark Pei
@email: duchungk7@gmail.com
"""

# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, 3:5].values


from sklearn.cluster import KMeans
wcss = []

# Calc K value
# Using the elbow method to find the optimal number of clusters
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i,
                    init= 'k-means++',
                    n_init= 10,
                    max_iter= 300,
                    random_state= 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)


plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.savefig('WCSS.png')
plt.show()

# Applying the K-means to the mall dataset
kmeans = KMeans(n_clusters = 5,
                    init= 'k-means++',
                    n_init= 10,
                    max_iter= 300,
                    random_state= 0)

y_kmeans = kmeans.fit_predict(X)


# Visualizing the Clusters
plt.scatter(X[y_kmeans ==0, 0], X[y_kmeans ==0, 1], s = 50, c = 'red', label = 'Cluster 0')
plt.scatter(X[y_kmeans ==1, 0], X[y_kmeans ==1, 1], s = 50, c = 'blue', label = 'Cluster 1')
plt.scatter(X[y_kmeans ==2, 0], X[y_kmeans ==2, 1], s = 50, c = 'green', label = 'Cluster 2')
plt.scatter(X[y_kmeans ==3, 0], X[y_kmeans ==3, 1], s = 50, c = 'cyan', label = 'Cluster 3')
plt.scatter(X[y_kmeans ==4, 0], X[y_kmeans ==4, 1], s = 50, c = 'magenta', label = 'Cluster 4')

plt.scatter(kmeans.cluster_centers_[:, 0],kmeans.cluster_centers_[:, 1], s = 200, c = 'yellow', label = 'Centroids')

plt.title('Clusters of clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.savefig('kmeans_result.png')
plt.show()

