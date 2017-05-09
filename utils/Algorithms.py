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
        training_samples = None if samples['training'] else np.array(samples['training'])
        testing_samples  = None if samples['testing']  else np.array(samples['testing'])
        return training_samples, testing_samples

    def unpack_labels(self, data):
        labels          = data['labels']
        training_labels = None if labels['training']  else np.array(labels['training'])
        testing_labels  = None if labels['testing']   else np.array(labels['testing'])
        return training_labels, testing_labels

    def unpack_ids(self, data):
        ids          = data['ids']
        training_ids = None if ids['training']  else np.array(ids['training'])
        testing_ids  = None if ids['testing']   else np.array(ids['testing'])
        return training_ids, testing_ids

    def pack_data(self, training_ids, testing_ids, training_samples, testing_samples, training_labels, testing_labels, predicted_labels = None, cluster_centers = None):
        data = {
            'ids':     {
                'training': training_ids, 
                'testing': testing_ids
            },
            'samples': {
                'training': training_samples, 
                'testing': testing_samples
            },
            'labels':  {
                'training': training_labels, 
                'testing': testing_labels
            },
            'predicting': predicted_labels, 
            'cluster_centers': cluster_centers
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
        strategy        = "mean" if not 'impute-strategy' in params else params['impute-strategy']
        axis            = 0
        imputer         = preprocessing.Imputer(missing_values = missing_values, strategy = strategy, axis = axis)

        # TODO: Fix label is not defined
        if training_samples is not None:
            unique_tag = np.unique(labels)
            for tag in uniquetag:
                tag_indexes = np.where(labels == tag)
                tag_indexes.shape = 1, -1
                training_sample = training_samples[tag_indexes]
                imputer.fit(training_sample)
                training_sample = imputer.transform(training_sample)
                training_samples[tag_indexes] = training_sample
        if testing_samples is not None:
            imputer.fit(testing_samples)
            testing_samples = imputer.transform(testing_samples)

        return self.pack_data(training_ids, testing_ids, training_samples, testing_samples, training_labels, testing_labels)

    def normalization(self, data, params):
        training_ids, testing_ids         = self.unpack_ids(data)
        training_samples, testing_samples = self.unpack_samples(data)
        training_labels, testing_labels   = self.unpack_labels(data)
        axis                              = 0
        
        if training_samples is not None:
            training_samples = preprocessing.scale(training_samples, axis = axis)
        if testing_samples is not None:
            testing_samples = preprocessing.scale(testing_samples, axis = axis)

        return self.pack_data(training_ids, testing_ids, training_samples, testing_samples, training_labels, testing_labels)

    def k_nearest_neighbors(self, data, params):
        training_ids, testing_ids         = self.unpack_ids(data)
        training_samples, testing_samples = self.unpack_samples(data)
        training_labels, testing_labels   = self.unpack_labels(data)
        
        n_neighbors = 5 if not 'n-neighbors' in params else params['n-neighbors']
        weights     = 'uniform' if not 'weights' in  params else params['weights']
        metric      = 'minkowski' if not 'metric' in params else params['metric']
        knn_clf     = neighbors.KNeighborsClassifier(n_neighbors = n_neighbors, weights = weights, metric = metric)
        knn_clf.fit(training_samples,training_labels)

        predicted_labels = knn_clf.predict(testing_samples)
        return self.pack_data(training_ids,testing_ids, training_samples, testing_samples,\
                training_labels, testing_labels, predicted_labels)

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
        
        predicted_labels = svm_clf.predict(testing_samples)
        return self.pack_data(training_ids, testing_ids, training_samples, testing_samples, \
                training_labels, testing_labels, predicted_labels)

    def random_forest(self, data, params):
        training_ids, testing_ids         = self.unpack_ids(data)
        training_samples, testing_samples = self.unpack_samples(data)
        training_labels, testing_labels   = self.unpack_labels(data)
        
        n_estimators      = 10 if not 'n-estimators' in params else params['n-estimators']
        criterion         = "gini" if not 'criterion' in params else params['criterion']
        random_forest_clf = ensemble.RandomForestClassifier(n_estimators = n_estimators, criterion = criterion)
        random_forest_clf.fit(training_samples, training_labels)
        
        predicted_labels = random_forest_clf.predict(testing_samples)
        return self.pack_data(training_ids, testing_ids, training_samples, testing_samples, \
                training_labels, testing_labels, predicted_labels)

    def k_means(self, data, params):
        training_ids, testing_ids         = self.unpack_ids(data)
        training_samples, testing_samples = self.unpack_samples(data)
        training_labels, testing_labels   = self.unpack_labels(data)
        
        n_clusters     = 8 if not 'n-clusters' in params else params['n-clusters']
        kmeans_cluster = cluster.KMeans(n_clusters = n_clusters)
        kmeans_cluster.fit(training_samples)
        
        predicted_labels = kmeans_cluster.labels_
        return self.pack_data(training_ids, testing_ids, training_samples, testing_samples, \
                training_labels, testing_labels, predicted_labels)

    def hierarchical_cluster(self, data, params):
        training_ids, testing_ids         = self.unpack_ids(data)
        training_samples, testing_samples = self.unpack_samples(data)
        training_labels, testing_labels   = self.unpack_labels(data)
        
        n_clusters = 2 if not 'n-clusters' in params else params['n-clusters']
        h_cluster  = cluster.AgglomerativeClustering(n_clusters = n_clusters)
        h_cluster.fit(training_samples)
        predicted_labels = h_cluster.labels_
        
        return self.pack_data(training_ids, testing_ids, training_samples, testing_samples, \
                training_labels, testing_labels, predicted_labels)

    def pca(self, data, params):
        training_ids, testing_ids           = self.unpack_ids(data)
        training_samples, testing_samples   = self.unpack_samples(data)
        training_labels, testing_labels     = self.unpack_labels(data)

        print training_ids

        n_components     = 100 if not 'n-components' in params else params['n-components']
        pca              = decomposition.PCA(n_components = n_components)
        high_data        = np.concatenate((training_samples, testing_samples),axis = 0)
        low_data         = pca.fit(high_data).transform(high_data)
        training_samples = low_data[0:training_samples.shape[0]]
        testing_samples  = low_data[training_samples.shape[0]:training_samples.shape[0] + testing_samples.shape[0]]

        return self.pack_data(training_ids, testing_ids, training_samples, testing_samples, training_labels, testing_labels)

    def random_projection(self, data, params):
        training_ids, testing_ids         = self.unpack_ids(data)
        training_samples, testing_samples = self.unpack_samples(data)
        training_labels, testing_labels   = self.unpack_labels(data)
        
        n_components     = 100 if not 'n-components' in params else params['n-components']
        rp               = random_projection.SparseRandomProjection(n_components = n_components)
        training_samples = rp.fit_transform(training_samples)
        testing_samples  = rp.fit_transform(testing_samples)
        return self.pack_data(training_ids, testing_ids, training_samples, testing_samples, training_labels, testing_labels)

    def feature_selection(self, data, params):
        training_ids, testing_ids         = self.unpack_ids(data)
        training_samples, testing_samples = self.unpack_samples(data)
        training_labels, testing_labels   = self.unpack_labels(data)
        
        score_func       = 'chi2' if not 'select-strategy' in params else params['select-strategy']
        k                = 100 if not 'n-components' in params else params['n-components']
        high_data        = np.concatenate((training_samples, testing_samples), axis = 0)
        low_data         = feature_selection.SelectKBest(score_func = score_func, k = k).fit_transform(high_data)
        training_samples = low_data[0:training_samples.shape[0]]
        testing_samples  = low_data[training_samples.shape[0]:training_samples.shape[0] + testing_samples.shape[0]]

        return self.pack_data(training_ids, testing_ids, training_samples, testing_samples, training_labels, testing_labels)

    def tsne(self, data, params):
        training_ids, testing_ids         = self.unpack_ids(data)
        training_samples, testing_samples = self.unpack_samples(data)
        training_labels, testing_labels   = self.unpack_labels(data)
        
        n_components     = 2
        n_iter           = 2500 if not 'n-iterations' in params else params['n-iterations']
        high_data        = np.concatenate((training_samples, testing_samples), axis = 0)
        low_data         = manifold.TSNE(n_components = n_components, n_iter = n_iter).fit_transform(high_data)
        training_samples = low_data[0:training_samples.shape[0]]
        testing_samples  = low_data[training_samples.shape[0]:training_samples.shape[0] + testing_samples.shape[0]]

        return self.pack_data(training_ids, testing_ids, training_samples, testing_samples, training_labels, testing_labels)

    def mds(self, data, params):
        training_ids, testing_ids         = self.unpack_ids(data)
        training_samples, testing_samples = self.unpack_samples(data)
        training_labels, testing_labels   = self.unpack_labels(data)
       
        n_components     = 2
        n_init           = 1
        max_iter         = 300 if not 'max-iterations' in params else params['max-iterations']
        high_data        = np.concatenate((training_samples, testing_samples), axis = 0)
        low_data         = manifold.MDS(n_components = n_components, max_iter = max_iter, n_init = n_init).fit_transform(high_data)
        training_samples = low_data[0:training_samples.shape[0]]
        testing_samples  = low_data[training_samples.shape[0]:training_samples.shape[0] + testing_samples.shape[0]]

        return self.pack_data(training_ids, testing_ids, training_samples, testing_samples, training_labels, testing_labels)
