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


class Algorithm(object):
    def unpack_samples(self, data):
        samples          = data['samples']
        training_samples = None if samples['training'] is None else np.array(samples['training'])
        testing_samples  = None if samples['testing']  is None else np.array(samples['testing'])
        return training_samples, testing_samples

    def unpack_labels(self, data):
        labels          = data['labels']
        training_labels = None if labels['training']  is None else np.array(labels['training'])
        testing_labels  = None if labels['testing']   is None else np.array(labels['testing'])
        return training_labels, testing_labels

    def unpack_ids(self, data):
        ids          = data['ids']
        training_ids = None if ids['training']  is None else np.array(ids['training'])
        testing_ids  = None if ids['testing']   is None else np.array(ids['testing'])
        return training_ids, testing_ids

    def unpack_params(self, data):
        params = data['params']
        return params

    def pack_data(self, training_ids, testing_ids, training_samples, testing_samples, training_labels, testing_labels, predicted_labels = None, cluster_centers = None):
        data = { \
            'ids':     {'training': training_ids, 'testing': testing_ids}, \
            'samples': {'training': training_samples, 'testing': testing_samples}, \
            'labels':  {'training': training_labels, 'testing': testing_labels, \
            'predicting': predicted_labels, 'cluster_centers': cluster_centers} \
        }
        return data

    def impute_missing_values(self, data):
        training_ids, testing_ids         = unpack_ids(data)
        training_samples, testing_samples = unpack_samples(data)
        training_labels, testing_labels   = unpack_labels(data)
        
        params          = unpack_params(data)
        missing_values  = "NaN" if params['missing_values'] is None else params['missing_values']
        strategy        = "mean" if params['strategy'] is None else params['strategy']
        axis            = 0
        imputer         = preprocessing.Imputer(missing_values = missing_values, strategy = strategy, axis = axis)

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

        return pack_data(training_ids, testing_ids, training_samples, testing_samples, training_labels, testing_labels)

    def normalization(self, data):
        training_ids, testing_ids         = unpack_ids(data)
        training_samples, testing_samples = unpack_samples(data)
        training_labels, testing_labels   = unpack_labels(data)
        axis                              = 0
        
        if training_samples is not None:
            training_samples = preprocessing.scale(training_samples, axis = axis)
        if testing_samples is not None:
            testing_samples = preprocessing.scale(testing_samples, axis = axis)

        return pack_data(training_ids, testing_ids, training_samples, testing_samples, training_labels, testing_labels)

    def k_nearest_neighbors(self, data):
        training_ids, testing_ids         = unpack_ids(data)
        training_samples, testing_samples = unpack_samples(data)
        training_labels, testing_labels   = unpack_labels(data)
        
        params      = unpack_params(data)
        n_neighbors = 5 if params['n_neighbors'] is None else params['n_neighbors']
        weights     = 'uniform' if params['weights'] is None else params['weights']
        metric      = 'minkowski' if params['metric'] is None else params['metric']
        
        knn_clf     = neighbors.KNeighborsClassifier(n_neighbors = n_neighbors, weights = weights, metric = metric)
        knn_clf.fit(training_samples,training_labels)

        predicted_labels = knn_clf.predict(testing_samples)
        return pack_data(training_ids,testing_ids, training_samples, testing_samples,\
                training_labels, testing_labels, predicted_labels)

    def svm(self, data):
        training_ids, testing_ids         = unpack_ids(data)
        training_samples, testing_samples = unpack_samples(data)
        training_labels, testing_labels   = unpack_labels(data)

        params       = unpack_params(data)
        kernel       = 'rbf' if params['kernel'] is None else params['kernel']
        C            = 1.0 if params['C'] is None else params['C']
        degree       = 3 if params['degree'] is None else params['degree']
        gamma        = 'auto' if params['gamma'] is None else params['gamma']
        
        svm_clf = svm.SVC(C=C, kernel = kernel, degree = degree, gamma = gamma)
        svm_clf.fit(training_samples, training_labels)
        
        predicted_labels = svm_clf.predict(testing_samples)
        return pack_data(training_ids, testing_ids, training_samples, testing_samples, \
                training_labels, testing_labels, predicted_labels)

    def random_forest(self, data):
        training_ids, testing_ids         = unpack_ids(data)
        training_samples, testing_samples = unpack_samples(data)
        training_labels, testing_labels   = unpack_labels(data)
        
        params       = unpack_params(data)
        n_estimators = 10 if params['n_estimators'] is None else params['n_estimators']
        criterion    = "gini" if params['criterion'] is None else params['criterion']
        
        random_forest_clf = ensemble.RandomForestClassifier(n_estimators = n_estimators, criterion = criterion)
        random_forest_clf.fit(training_samples, training_labels)
        
        predicted_labels = random_forest_clf.predict(testing_samples)
        return pack_data(training_ids, testing_ids, training_samples, testing_samples, \
                training_labels, testing_labels, predicted_labels)

    def k_means(self, data):
        training_ids, testing_ids         = unpack_ids(data)
        training_samples, testing_samples = unpack_samples(data)
        training_labels, testing_labels   = unpack_labels(data)
        
        params       = unpack_params(data)
        n_clusters   = 8 if params['n_cluster'] is None else params['n_cluster']

        kmeans_cluster = cluster.KMeans(n_clusters = n_clusters)
        kmeans_cluster.fit(training_samples)
        
        predicted_labels = kmeans_cluster.labels_
        return pack_data(training_ids, testing_ids, training_samples, testing_samples, \
                training_labels, testing_labels, predicted_labels)

    def hierarchical_cluster(self, data):
        training_ids, testing_ids         = unpack_ids(data)
        training_samples, testing_samples = unpack_samples(data)
        training_labels, testing_labels   = unpack_labels(data)
        
        params     = unpack_params(data)
        n_clusters = 2 if params['n_cluster'] is None else params['n_cluster']
        
        h_cluster = cluster.AgglomerativeClustering(n_clusters = n_clusters)
        h_cluster.fit(training_samples)
        predicted_labels = h_cluster.labels_
        
        return pack_data(training_ids, testing_ids, training_samples, testing_samples, \
                training_labels, testing_labels, predicted_labels)

    def pca(self, data):
        training_ids, testing_ids           = unpack_ids(data)
        training_samples, testing_samples   = unpack_samples(data)
        training_labels, testing_labels     = unpack_labels(data)

        params       = unpack_params(data)
        n_components = params['n_components']

        pca              = decomposition.PCA(n_components = n_components)
        high_data        = np.concatenate((training_samples, testing_samples),axis = 0)
        low_data         = pca.fit(high_data).transform(high_data)
        training_samples = low_data[0:training_samples.shape[0]]
        testing_samples  = low_data[training_samples.shape[0]:training_samples.shape[0] + testing_samples.shape[0]]

        return pack_data(training_ids, testing_ids, training_samples, testing_samples, training_labels, testing_labels)

    def random_projection(self, data):
        training_ids, testing_ids         = unpack_ids(data)
        training_samples, testing_samples = unpack_samples(data)
        training_labels, testing_labels   = unpack_labels(data)
        
        params           = unpack_params(data)
        n_components     = params['n_components']

        rp               = random_projection.SparseRandomProjection(n_components = n_components)
        training_samples = rp.fit_transform(training_samples)
        testing_samples  = rp.fit_transform(testing_samples)
        return pack_data(training_ids, testing_ids, training_samples, testing_samples, training_labels, testing_labels)

    def feature_selection(self, data):
        training_ids, testing_ids         = unpack_ids(data)
        training_samples, testing_samples = unpack_samples(data)
        training_labels, testing_labels   = unpack_labels(data)
        
        params     = unpack_params(data)
        score_func = params['score_func']
        k          = params['k'] if params['k'] >= 1 else k = int(float(params['k'])*training_samples.shape[1])

        high_data  = np.concatenate((training_samples, testing_samples), axis = 0)
        low_data         = feature_selection.SelectKBest(score_func = score_func, k = k).fit_transform(high_data)
        training_samples = low_data[0:training_samples.shape[0]]
        testing_samples  = low_data[training_samples.shape[0]:training_samples.shape[0] + testing_samples.shape[0]]

        return pack_data(training_ids, testing_ids, training_samples, testing_samples, training_labels, testing_labels)

    def tsne(self, data):
        training_ids, testing_ids         = unpack_ids(data)
        training_samples, testing_samples = unpack_samples(data)
        training_labels, testing_labels   = unpack_labels(data)
        
        params       = unpack_params(data)
        n_components = 2
        n_iter       = 1000 if params['n_iter'] is None else params['n_iter']
        
        high_data        = np.concatenate((training_samples, testing_samples), axis = 0)
        low_data         = manifold.TSNE(n_components = n_components, n_iter = n_iter).fit_transform(high_data)
        training_samples = low_data[0:training_samples.shape[0]]
        testing_samples  = low_data[training_samples.shape[0]:training_samples.shape[0] + testing_samples.shape[0]]

        return pack_data(training_ids, testing_ids, training_samples, testing_samples, training_labels, testing_labels)

    def mds(self, data):
        training_ids, testing_ids         = unpack_ids(data)
        training_samples, testing_samples = unpack_samples(data)
        training_labels, testing_labels   = unpack_labels(data)
       
        params       = unpack_params(data)
        n_components = 2
        n_init       = 1
        max_iter     = 300 if params['max_iter'] is None else params['max_iter']

        high_data        = np.concatenate((training_samples, testing_samples), axis = 0)
        low_data         = manifold.MDS(n_components = n_components, max_iter = max_iter, n_init = n_init).fit_transform(high_data)
        training_samples = low_data[0:training_samples.shape[0]]
        testing_samples  = low_data[training_samples.shape[0]:training_samples.shape[0] + testing_samples.shape[0]]

        return pack_data(training_ids, testing_ids, training_samples, testing_samples, training_labels, testing_labels)
