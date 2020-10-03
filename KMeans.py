'''

Practice Exercise: K-means Clustering
Watts Dietrich
Oct 3 2020

A very simple example of k-means clustering using sklearn.
The program creates 10 clusters with randomized initial centroid positions and prints a series of different
accuracy scores to assess the resulting model.

'''


import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

# load the digits dataset from sklearn
digits = load_digits()

# scale features down to range from -1 to 1
data = scale(digits.data)

y = digits.target

# set number of centroids. could hard-code a number, but here we dynamically scale k to the dataset
k = len(np.unique(y))
samples, features = data.shape #data.shape will return 2 variables (# of rows and columns), so we assign to 2 variables

# function for printing a list of different accuracy scores
def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))

# train the model and call the function above to show the accuracy scores
clf = KMeans(n_clusters=k, init='random', n_init=10)
bench_k_means(clf, "scores", data)
