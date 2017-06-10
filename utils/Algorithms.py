#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

from sklearn import preprocessing       # Imputer, Scale
from sklearn import neighbors           # KNeighborsClassifier
from sklearn import svm                 # SVC
from sklearn import ensemble            # RandomForestClassifier, AdaBoostClassifier
from sklearn import cluster             # AgglomerativeClustering, KMeans
from sklearn import decomposition       # PCA
from sklearn import random_projection
from sklearn import feature_selection
from sklearn import manifold            # TSNE, MDS

class Algorithms(object):
    def unpack_samples(self, data):
        samples          = data['samples']
        training_samples = np.array(samples['training']) if 'training' in samples else None
        testing_samples  = np.array(samples['testing']) if 'testing' in samples else None
        return training_samples, testing_samples

    def unpack_labels(self, data):
        labels          = data['labels']
        training_labels = np.array(labels['training']) if 'training' in labels else None
        testing_labels  = np.array(labels['testing']) if 'testing' in labels else None
        return training_labels, testing_labels

    def unpack_ids(self, data):
        ids          = data['ids']
        training_ids = np.array(ids['training']) if 'training' in ids else None
        testing_ids  = np.array(ids['testing']) if 'testing' in ids else None
        return training_ids, testing_ids

    def pack_data(self, training_ids, testing_ids, training_samples, testing_samples, training_labels, testing_labels, training_predicted_labels = None, testing_predicted_labels = None, cluster_centers = None, nearest_neighbors = None):
        training_ids     = np.ndarray.tolist(training_ids) if isinstance(training_ids, np.ndarray) else None
        testing_ids      = np.ndarray.tolist(testing_ids) if isinstance(testing_ids, np.ndarray) else None
        training_samples = np.ndarray.tolist(training_samples) if isinstance(training_samples, np.ndarray) else None
        testing_samples  = np.ndarray.tolist(testing_samples) if isinstance(testing_samples, np.ndarray) else None
        training_labels  = np.ndarray.tolist(training_labels) if isinstance(training_labels, np.ndarray) else None
        testing_labels   = np.ndarray.tolist(testing_labels) if isinstance(testing_labels, np.ndarray) else None
        training_predicted_labels = np.ndarray.tolist(training_predicted_labels) if isinstance(training_predicted_labels, np.ndarray) else None
        testing_predicted_labels = np.ndarray.tolist(testing_predicted_labels) if isinstance(testing_predicted_labels, np.ndarray) else None
        cluster_centers  = np.ndarray.tolist(cluster_centers) if isinstance(cluster_centers, np.ndarray) else None
        nearest_neighbors = np.ndarray.tolist(nearest_neighbors) if isinstance(nearest_neighbors, np.ndarray) else None

        data = {
            'ids':     {
                'training': training_ids, 
                'testing' : testing_ids
            },
            'samples': {
                'training': training_samples, 
                'testing' : testing_samples
            },
            'labels':  {
                'training': training_labels, 
                'testing' : testing_labels
            },
            'predicting':{
                'training': training_predicted_labels,
                'testing' : testing_predicted_labels
            }, 
            'cluster_centers': cluster_centers,
            'nearest_neighbors': nearest_neighbors
        }
        return data

    def get_algorithm(self, algorithm_name):
        algorithm = getattr(self, algorithm_name, None)
        return algorithm

    def impute_missing_values(self, data, params):
        training_ids, testing_ids         = self.unpack_ids(data)
        training_samples, testing_samples = self.unpack_samples(data)
        training_labels, testing_labels   = self.unpack_labels(data)

        missing_values  = "NaN" if not 'missing-values' in params else params['missing-values']
        missing_values  = "NaN" if missing_values == "" else missing_values
        strategy        = "mean" if not 'impute-strategy' in params else params['impute-strategy']
        axis            = 0
        imputer         = preprocessing.Imputer(missing_values = missing_values, strategy = strategy, axis = axis)

        if training_samples.any():
            imputer.fit_transform(training_samples)
            # unique_tag = np.unique(training_labels)
            # for tag in unique_tag:
            #     tag_indexes = np.where(training_labels == tag)
            #     tag_indexes.shape = 1, -1
            #     training_sample = training_samples[tag_indexes]
            #     imputer.fit(training_sample)
            #     training_sample = imputer.transform(training_sample)
            #     training_samples[tag_indexes] = training_sample
        if testing_samples.any():
            imputer.fit_transform(testing_samples)
        return self.pack_data(training_ids = training_ids, testing_ids = testing_ids, training_samples = training_samples, testing_samples = testing_samples, training_labels = training_labels, testing_labels = testing_labels)

    def zscore_normalization(self, data, params):
        training_ids, testing_ids         = self.unpack_ids(data)
        training_samples, testing_samples = self.unpack_samples(data)
        training_labels, testing_labels   = self.unpack_labels(data)
        axis                              = 0
        if training_samples.any():
            training_samples = preprocessing.scale(training_samples, axis = axis)
        if testing_samples.any():
            testing_samples = preprocessing.scale(testing_samples, axis = axis)

        return self.pack_data(training_ids = training_ids, testing_ids = testing_ids, training_samples = training_samples, testing_samples = testing_samples, training_labels = training_labels, testing_labels = testing_labels)

    def zero_centered_normalization(self, data, params):
        training_ids, testing_ids         = self.unpack_ids(data)
        training_samples, testing_samples = self.unpack_samples(data)
        training_labels, testing_labels   = self.unpack_labels(data)
        scaler = preprocessing.StandardScaler(with_std = False)
        if training_samples.any():
            training_samples = scaler.fit_transform(training_samples)
        if testing_samples.any():
            testing_samples = scaler.fit_transform(testing_samples)

        return self.pack_data(training_ids = training_ids, testing_ids = testing_ids, training_samples = training_samples, testing_samples = testing_samples, training_labels = training_labels, testing_labels = testing_labels)


    def min_max_normalization(self, data, params):
        training_ids, testing_ids         = self.unpack_ids(data)
        training_samples, testing_samples = self.unpack_samples(data)
        training_labels, testing_labels   = self.unpack_labels(data)
        scaler = preprocessing.MinMaxScaler()
        if training_samples.any():
            training_samples = scaler.fit_transform(training_samples)
        if testing_samples.any():
            testing_samples = scaler.fit_transform(testing_samples)

        return self.pack_data(training_ids = training_ids, testing_ids = testing_ids, training_samples = training_samples, testing_samples = testing_samples, training_labels = training_labels, testing_labels = testing_labels)

    def max_abs_normalization(self, data, params):
        training_ids, testing_ids         = self.unpack_ids(data)
        training_samples, testing_samples = self.unpack_samples(data)
        training_labels, testing_labels   = self.unpack_labels(data)
        scaler = preprocessing.MaxAbsScaler()
        if training_samples.any():
            training_samples = scaler.fit_transform(training_samples)
        if testing_samples.any():
            testing_samples = scaler.fit_transform(testing_samples)

        return self.pack_data(training_ids = training_ids, testing_ids = testing_ids, training_samples = training_samples, testing_samples = testing_samples, training_labels = training_labels, testing_labels = testing_labels)

    def median_centered_normalization(self, data, params):
        training_ids, testing_ids         = self.unpack_ids(data)
        training_samples, testing_samples = self.unpack_samples(data)
        training_labels, testing_labels   = self.unpack_labels(data)
        scaler = preprocessing.RobustScaler()
        if training_samples.any():
            training_samples = scaler.fit_transform(training_samples)
        if testing_samples.any():
            testing_samples = scaler.fit_transform(testing_samples)

        return self.pack_data(training_ids = training_ids, testing_ids = testing_ids, training_samples = training_samples, testing_samples = testing_samples, training_labels = training_labels, testing_labels = testing_labels)

    def knn(self, data, params):
        training_ids, testing_ids         = self.unpack_ids(data)
        training_samples, testing_samples = self.unpack_samples(data)
        training_labels, testing_labels   = self.unpack_labels(data)
        
        n_neighbors = 5 if (not 'n-neighbors' in params or not params['n-neighbors']) else int(params['n-neighbors'])
        weights     = 'uniform' if not 'weights' in  params else params['weights']
        metric      = 'minkowski' if not 'metric' in params else params['metric']
        knn_clf     = neighbors.KNeighborsClassifier(n_neighbors = n_neighbors, weights = weights, metric = metric)
        knn_clf.fit(training_samples,training_labels)
        if testing_samples.any():
            testing_predicted_labels = knn_clf.predict(testing_samples)
        return self.pack_data(training_ids = training_ids, testing_ids = testing_ids, training_samples = training_samples, testing_samples = testing_samples,\
                training_labels = training_labels, testing_labels = testing_labels, testing_predicted_labels = testing_predicted_labels)

    def svm(self, data, params):
        training_ids, testing_ids         = self.unpack_ids(data)
        training_samples, testing_samples = self.unpack_samples(data)
        training_labels, testing_labels   = self.unpack_labels(data)

        kernel       = 'rbf' if not 'kernel' in params else params['kernel']
        C            = 1.0 if not 'penalty-parameter' in params else params['penalty-parameter']
        degree       = 3 if not 'degree' in params else params['degree']
        gamma        = 'auto' if not 'gamma' in params else params['gamma']
        svm_clf      = svm.SVC(C=C, kernel = kernel, degree = degree, gamma = gamma)
        svm_clf.fit(training_samples, training_labels)
        if testing_samples.any():
            testing_predicted_labels = svm_clf.predict(testing_samples)
        return self.pack_data(training_ids = training_ids, testing_ids = testing_ids, training_samples = training_samples, testing_samples = testing_samples, \
                training_labels = training_labels, testing_labels = testing_labels, testing_predicted_labels = testing_predicted_labels)

    def random_forest(self, data, params):
        training_ids, testing_ids         = self.unpack_ids(data)
        training_samples, testing_samples = self.unpack_samples(data)
        training_labels, testing_labels   = self.unpack_labels(data)

        n_estimators      = 10 if (not 'n-estimators' in params or not params['n-estimators'])else int(params['n-estimators'])
        criterion         = "gini" if not 'criterion' in params else params['criterion']
        random_forest_clf = ensemble.RandomForestClassifier(n_estimators = n_estimators, criterion = criterion)
        random_forest_clf.fit(training_samples, training_labels)
        if testing_samples.any():
            testing_predicted_labels = random_forest_clf.predict(testing_samples)
        return self.pack_data(training_ids = training_ids, testing_ids = testing_ids, training_samples = training_samples, testing_samples = testing_samples, \
                training_labels = training_labels, testing_labels = testing_labels, testing_predicted_labels = testing_predicted_labels)

    def kmeans(self, data, params):
        training_ids, testing_ids         = self.unpack_ids(data)
        training_samples, testing_samples = self.unpack_samples(data)
        training_labels, testing_labels   = self.unpack_labels(data)

        n_clusters     = 5 if (not 'n-clusters' in params or not params['n-clusters']) else int(params['n-clusters'])
        kmeans_cluster = cluster.KMeans(n_clusters = n_clusters)
        kmeans_cluster.fit(training_samples)
        training_predicted_labels = kmeans_cluster.labels_
        cluster_centers  = kmeans_cluster.cluster_centers_
        n_neighbors = 11
        neighborhood = neighbors.NearestNeighbors(n_neighbors = n_neighbors)
        neighborhood.fit(training_samples)
        neighbor_indices = neighborhood.kneighbors(training_samples, return_distance=False)
        nearest_neighbors = neighbor_indices[:,1:]
        return self.pack_data(training_ids = training_ids, testing_ids = testing_ids, training_samples = training_samples, testing_samples = testing_samples, \
                training_labels = training_labels, testing_labels = testing_labels, training_predicted_labels = training_predicted_labels, cluster_centers = cluster_centers, nearest_neighbors = nearest_neighbors)

    def hierarchical_cluster(self, data, params):
        training_ids, testing_ids         = self.unpack_ids(data)
        training_samples, testing_samples = self.unpack_samples(data)
        training_labels, testing_labels   = self.unpack_labels(data)

        n_clusters = 2 if not 'n-clusters' in params else int(params['n-clusters'])
        linkage  = 'ward' if not 'linkage' in params else params['linkage']
        h_cluster  = cluster.AgglomerativeClustering(n_clusters = n_clusters, linkage = linkage)
        h_cluster.fit(training_samples)
        training_predicted_labels = h_cluster.labels_
        childrens = sorted(h_cluster.children_, key = lambda x: x[0])
        
        return self.pack_data(training_ids = training_ids, testing_ids = testing_ids, training_samples = training_samples, testing_samples = testing_samples, \
                training_labels = training_labels, testing_labels = testing_labels, training_predicted_labels = training_predicted_labels)

    def pca(self, data, params):
        training_ids, testing_ids           = self.unpack_ids(data)
        training_samples, testing_samples   = self.unpack_samples(data)
        training_labels, testing_labels     = self.unpack_labels(data)

        n_components     = 100 if (not 'n-components' in params or not params['n-components']) else int(params['n-components'])
        pca              = decomposition.PCA(n_components = n_components)
        high_data        = np.concatenate((training_samples, testing_samples), 0) if testing_samples.any() else training_samples
        low_data         = pca.fit(high_data).transform(high_data)
        training_samples = low_data[0:training_samples.shape[0]]
        if testing_samples.any():
            testing_samples  = low_data[training_samples.shape[0]:(training_samples.shape[0] + testing_samples.shape[0])]

        return self.pack_data(training_ids = training_ids, testing_ids = testing_ids, training_samples = training_samples, testing_samples = testing_samples, training_labels = training_labels, testing_labels = testing_labels)

    def random_projection(self, data, params):
        training_ids, testing_ids         = self.unpack_ids(data)
        training_samples, testing_samples = self.unpack_samples(data)
        training_labels, testing_labels   = self.unpack_labels(data)
        
        n_components     = 100 if not 'n-components' in params else int(params['n-components'])
        rp               = random_projection.SparseRandomProjection(n_components = n_components)
        training_samples = rp.fit_transform(training_samples)
        testing_samples  = rp.fit_transform(testing_samples) if testing_samples.any() else testing_samples
        return self.pack_data(training_ids = training_ids, testing_ids = testing_ids, training_samples = training_samples, testing_samples = testing_samples, training_labels = training_labels, testing_labels = testing_labels)

    def feature_selection(self, data, params):
        training_ids, testing_ids         = self.unpack_ids(data)
        training_samples, testing_samples = self.unpack_samples(data)
        training_labels, testing_labels   = self.unpack_labels(data)
        score_func       = 'chi2' if not 'select-strategy' in params else params['select-strategy']
        k                = 100 if not 'n-components' in params else int(params['n-components'])
        high_data        = np.concatenate((training_samples, testing_samples), axis = 0) if testing_samples.any() else training_samples
        selector         = feature_selection.SelectKBest(score_func = score_func, k = k)
        low_data         = selector.fit_transform(high_data)
        training_samples = low_data[0:training_samples.shape[0]]
        if testing_samples.any():
            testing_samples  = low_data[training_samples.shape[0]:(training_samples.shape[0] + testing_samples.shape[0])]
        return self.pack_data(training_ids = training_ids, testing_ids = testing_ids, training_samples = training_samples, testing_samples = testing_samples, training_labels = training_labels, testing_labels = testing_labels)

    def tsne(self, data, params):
        training_ids, testing_ids         = self.unpack_ids(data)
        training_samples, testing_samples = self.unpack_samples(data)
        training_labels, testing_labels   = self.unpack_labels(data)
        
        n_components     = 2
        n_iter           = 2500 if not 'n-iterations' in params else int(params['n-iterations'])
        high_data        = np.concatenate((training_samples, testing_samples), axis = 0) if testing_samples.any() else training_samples
        low_data         = manifold.TSNE(n_components = n_components, n_iter = n_iter).fit_transform(high_data)
        training_samples = low_data[0:training_samples.shape[0]]
        if testing_samples.any():
            testing_samples  = low_data[training_samples.shape[0]:(training_samples.shape[0] + testing_samples.shape[0])]

        return self.pack_data(training_ids = training_ids, testing_ids = testing_ids, training_samples = training_samples, testing_samples = testing_samples, training_labels = training_labels, testing_labels = testing_labels)

    def mds(self, data, params):
        training_ids, testing_ids         = self.unpack_ids(data)
        training_samples, testing_samples = self.unpack_samples(data)
        training_labels, testing_labels   = self.unpack_labels(data)
       
        n_components     = 2
        n_init           = 1
        max_iter         = 300 if not 'max-iterations' in params else int(params['max-iterations'])
        high_data        = np.concatenate((training_samples, testing_samples), axis = 0) if testing_samples.any() else training_samples
        low_data         = manifold.MDS(n_components = n_components, max_iter = max_iter, n_init = n_init).fit_transform(high_data)
        training_samples = low_data[0:training_samples.shape[0]]
        if testing_samples.any():
            testing_samples  = low_data[training_samples.shape[0]:(training_samples.shape[0] + testing_samples.shape[0])]

        return self.pack_data(training_ids = training_ids, testing_ids = testing_ids, training_samples = training_samples, testing_samples = testing_samples, training_labels = training_labels, testing_labels = testing_labels)
