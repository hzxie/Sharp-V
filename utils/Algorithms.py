#!/usr/bin/python
# -*- coding: utf-8 -*-
import json
import numpy as np
from sklearn import preprocessing#Imputer,Scale
from sklearn import neighbors#KNeighborsClassifier
from sklearn import svm#SVC
from sklearn import ensemble#RandomForestClassifier,AdaBoostClassifier
from sklearn import cluster#AgglomerativeClustering,KMeans
from sklearn import decomposition#PCA,
from sklearn import random_projection
from sklearn import feature_selection
from sklearn import manifold#TSNE,MDS
from scipy.stats import ttest_ind#t-test


class Algorithm(object):
    def __init__(self, data):
        datapackage = json.loads(data)
        self.samples = datapackage['samples']
        self.labels = datapackage['labels']
        self.params = datapackage['params']

    def unpack_samples(self):
        training_samples = None if self.samples['training'] is None else np.array(self.samples['training'])
        testing_samples  = None if self.samples['testing']  is None else np.array(self.samples['testing'])
        return training_samples, testing_samples

    def unpack_labels(self):
        training_labels  = None if self.labels['training']  is None else np.array(self.labels['training'])
        testing_labels   = None if self.labels['testing']   is None else np.array(self.labels['testing'])
        return training_labels, testing_labels

    def pack_data(self, training_samples, testing_samples, training_labels, testing_labels, predicted_labels = None, cluster_centers = None):
        if predicted_labels is None:
            data = {'samples': {'training': training_samples, 'testing': testing_samples}, \
                    'labels': {'training': training_labels, 'testing': testing_labels}}
        elif predicted_labels is not None and cluster_centers is None:
            data = {'samples': {'training': training_samples, 'testing': testing_samples}, \
                    'labels': {'training': training_labels, 'testing': testing_labels, 'predicting': predicted_labels}}
        else:
            data = {'samples': {'training': training_samples, 'testing': testing_samples}, \
                    'labels': {'training': training_labels, 'testing': testing_labels, \
                    'predicting': predicted_labels, 'cluster_centers': cluster_centers}}
        data = json.dumps(data)
        return data

    def impute_missing_values(self):
        training_samples, testing_samples = unpack_samples()
        training_labels, testing_labels = unpack_labels()
        missing_values = self.params['missing_values']
        strategy = self.params['strategy']
        axis = 0
        imputer = preprocessing.Imputer(missing_values = missing_values, strategy = strategy, axis = axis)
        if training_samples is not None:
            unique_tag = np.unique(labels)
            for tag in uniquetag:
                tag_indexes = np.where(labels == tag)
                tag_indexes.shape = 1,-1
                training_sample = training_samples[tag_indexes]
                imputer.fit(training_sample)
                training_sample = imputer.transform(training_sample)
                training_samples[tag_indexes] = training_sample
        if testing_samples is not None:
            imputer.fit(testing_samples)
            testing_samples = imputer.transform(testing_samples)
        ret_data = pack_data(training_samples, testing_samples, training_labels, testing_labels)
        return ret_data

    def normalize_data(self):
        training_samples, testing_samples = unpack_samples()
        training_labels, testing_labels = unpack_labels()
        axis = 0
        if training_samples is not None:
            training_samples = preprocessing.scale(training_samples, axis = axis)
        if testing_samples is not None:
            testing_samples = preprocessing.scale(testing_samples, axis = axis)
        ret_data = pack_data(training_samples, testing_samples, training_labels, testing_labels)
        return ret_data

    def k_nearest_neighbors(self):
        training_samples, testing_samples = unpack_samples()
        training_labels, testing_labels = unpack_labels()
        n_neighbors = self.params['n_neighbors']
        weights = self.params['weights']
        metric = self.params['metric']
        knn_clf = neighbors.KNeighborsClassifier(n_neighbors = n_neighbors, weights = weights, metric = metric)
        knn_clf.fit(training_samples,training_labels)
        predicted_labels = knn_clf.predict(testing_samples)
        ret_data = pack_data(training_samples, testing_samples, training_labels, testing_labels, predicted_labels)
        return ret_data

    def SVM(self):
        training_samples, testing_samples = unpack_samples()
        training_labels, testing_labels = unpack_labels()
        kernel = self.params['kernel']
        svm_clf = svm.SVC(kernel = kernel)
        svm_clf.fit(training_samples, training_labels)
        predicted_labels = svm_clf.predict(testing_samples)
        ret_data = pack_data(training_samples, testing_samples, training_labels, testing_labels, predicted_labels)
        return ret_data

    def random_forest(self):
        training_samples, testing_samples = unpack_samples()
        training_labels, testing_labels = unpack_labels()
        n_estimators = self.params['n_estimators']
        criterion = self.params['criterion']
        random_state = 1
        random_forest_clf = ensemble.RandomForestClassifier(n_estimators = n_estimators, criterion = criterion, random_state = random_state)
        random_forest_clf.fit(training_samples, training_labels)
        predicted_labels = random_forest_clf.predict(testing_samples)
        ret_data = pack_data(training_samples, testing_samples, training_labels, testing_labels, predicted_labels)
        return ret_data

    def k_means(self):
        training_samples, testing_samples = unpack_samples()
        training_labels, testing_labels = unpack_labels()
        n_clusters = self.params['n_cluster']
        random_state = 1
        kmeans_cluster = cluster.KMeans(n_clusters = n_clusters, random_state = random_state)
        kmeans_cluster.fit(training_samples)
        predicted_labels = kmeans_cluster.labels_
        ret_data = pack_data(training_samples, testing_samples, training_labels, testing_labels, predicted_labels)
        return ret_data

    def hierarchical_cluster(self):
        training_samples, testing_samples = unpack_samples()
        training_labels, testing_labels = unpack_labels()
        n_clusters = self.params['n_cluster']
        h_cluster = cluster.AgglomerativeClustering(n_clusters = n_clusters)
        h_cluster.fit(training_samples)
        predicted_labels = h_cluster.labels_
        ret_data = pack_data(training_samples, testing_samples, training_labels, testing_labels, predicted_labels)
        return ret_data

    def pca(self):
        training_samples, testing_samples = unpack_samples()
        training_labels, testing_labels = unpack_labels()
        n_components = self.params['n_components']
        pca = decomposition.PCA(n_components = n_components)
        high_data = np.concatenate((training_samples, testing_samples),axis = 0)
        low_data = pca.fit(high_data).transform(high_data)
        training_samples = low_data[0:training_samples.shape[0]]
        testing_samples = low_data[training_samples.shape[0]:training_samples.shape[0]+testing_samples.shape[0]]
        ret_data = pack_data(training_samples, testing_samples, training_labels, testing_labels)
        return ret_data

    def random_projection(self):
        training_samples, testing_samples = unpack_samples()
        training_labels, testing_labels = unpack_labels()
        n_components = self.params['n_components']
        eps = self.params['eps']
        rp = random_projection.SparseRandomProjection(n_components = n_components, eps = eps)
        training_samples = rp.fit_transform(training_samples)
        testing_samples = rp.fit_transform(testing_samples)
        ret_data = pack_data(training_samples, testing_samples, training_labels, testing_labels)
        return ret_data

    def feature_selection(self):
        training_samples, testing_samples = unpack_samples()
        training_labels, testing_labels = unpack_labels()
        score_func = self.params['score_func']
        k = self.params['k'] if self.params['k'] >= 1 else k = int(float(self.params['k'])*training_samples.shape[1])
        high_data = np.concatenate((training_samples, testing_samples), axis = 0)
        low_data = feature_selection.SelectKBest(score_func = score_func, k = k).fit_transform(high_data)
        training_samples = low_data[0:training_samples.shape[0]]
        testing_samples = low_data[training_samples.shape[0]:training_samples.shape[0]+testing_samples.shape[0]]
        ret_data = pack_data(training_samples, testing_samples, training_labels, testing_labels)
        return ret_data

    def tsne(self):
        training_samples, testing_samples = unpack_samples()
        training_labels, testing_labels = unpack_labels()
        n_components = 2
        n_iter = self.params['n_iter']
        random_state = 1
        high_data = np.concatenate((training_samples, testing_samples), axis = 0)
        low_data = manifold.TSNE(n_components = n_components, n_iter = n_iter, random_state = random_state).fit_transform(high_data)
        training_samples = low_data[0:training_samples.shape[0]]
        testing_samples = low_data[training_samples.shape[0]:training_samples.shape[0]+testing_samples.shape[0]]
        ret_data = pack_data(training_samples, testing_samples, training_labels, testing_labels)
        return ret_data

    def mds(self):
        training_samples, testing_samples = unpack_samples()
        training_labels, testing_labels = unpack_labels()
        n_components = 2
        n_init = 1
        max_iter = self.params['max_iter']
        random_state = 1
        high_data = np.concatenate((training_samples, testing_samples), axis = 0)
        low_data = manifold.MDS(n_components = n_components, max_iter = max_iter, n_init = n_init, random_state = random_state).fit_transform(high_data)
        training_samples = low_data[0:training_samples.shape[0]]
        testing_samples = low_data[training_samples.shape[0]:training_samples.shape[0]+testing_samples.shape[0]]
        ret_data = pack_data(training_samples, testing_samples, training_labels, testing_labels)
        return ret_data
