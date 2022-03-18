#Part II

#a.


#Importing libraries
from Orange.data import Table
from sklearn.cluster import KMeans
from collections import Counter, defaultdict
import numpy as np

#Reading data set into Table
data_tab = Table('ionoshpere')

#Applying KMeans clustering with 2 clusters and 0 random-state on data_tab
kmeans_estimator = KMeans(n_clusters= 2, random_state = 0)
k_clusters = kmeans_estimator.fit(data_tab)

#Printing the cluster labels, centers and inertia
print(k_clusters.labels_)
print(k_clusters.cluster_centers_)
print(k_clusters.inertia_)

#Caluculating the number of instances in both clusters
print(Counter(k_clusters.labels_))

#Storing indexes of instances of both clusters
clusters_index = defaultdict(list)
for index, c  in enumerate(k_clusters.labels_):
    clusters_index[c].append(index)

#Counting the instances in each cluster by label 'g' and 'b'
cluster0_g= 0
cluster0_b= 0
cluster1_g= 0
cluster1_b= 0


for a in clusters_index[0]:
    if data_tab[a]["y"] == "g":
        cluster0_g = cluster0_g + 1
    elif data_tab[a]["y"] == "b":
        cluster0_b = cluster0_b + 1


for b in clusters_index[1]:
    if data_tab[b]["y"] == "g":
        cluster1_g = cluster1_g + 1
    elif data_tab[b]["y"] == "b":
        cluster1_b = cluster1_b + 1

#Defining an array to present the instances in form of matrix of clusers and labels
A = np.array([[cluster0_g, cluster0_b], [cluster1_g, cluster1_b]])
print(A)

