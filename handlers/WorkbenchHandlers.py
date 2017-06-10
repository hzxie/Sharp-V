#!/usr/bin/python
# -*- coding: utf-8 -*-

import logging

from os import makedirs
from os.path import isfile as file_exists
from os.path import join as join_path
from os.path import exists as path_exists
from tornado.escape import json_decode as load_json
from tornado.escape import json_encode as dump_json
from re import match

from handlers.BaseHandler import BaseHandler
from utils.Algorithms import Algorithms
from utils.DatasetParsers import DatasetParser
from utils.DatasetParsers import MetasetParser

class DatasetUploadHandler(BaseHandler):
    def initialize(self):
        self.dataset_parser = DatasetParser()

    def post(self):
        file         = self.request.files['file'][0]
        file_content = file['body']
        content_type = file['content_type']
        file_name    = file['filename']
        current_user = self.get_current_user()
        dataset_name = self.get_argument('datasetName')
        result       = self.is_file_acceptable(content_type, file_name, current_user, dataset_name)

        if result['isSuccessful']:
            base_folder  = join_path(self.application.settings['static_path'], 'uploads', current_user, dataset_name)
            file_path    = join_path(base_folder, file_name)

            if not path_exists(base_folder):
                makedirs(base_folder)

            file = open(file_path, 'w')
            file.write(file_content)
            file.close()

            file_stats = self.dataset_parser.get_file_lines_columns(file_path)
            result['fileInfo']            = dict()
            result['fileInfo']['lines']   = file_stats[0]
            result['fileInfo']['columns'] = file_stats[1]
            logging.info('User [Username=%s] uploaded new file [%s] at %s.' % \
                (current_user, file_path, self.get_user_ip_addr()))

        self.write(dump_json(result))

    def is_file_acceptable(self, content_type, file_name, current_user, dataset_name):
        result = {
            'isContentTypeLegal':   True,
            'isDatasetNameEmpty':   dataset_name == '',
            'isDatasetNameLegal':   self.is_file_name_legal(dataset_name),
            'isDatasetNameExists':  False,
            'isFileNameLegal':      self.is_file_name_legal(file_name),
            'isFileNameExists':     False,
        }
        result['isSuccessful'] = result['isContentTypeLegal'] and not result['isDatasetNameEmpty'] and \
                                 result['isDatasetNameLegal'] and not result['isDatasetNameExists'] and \
                                 result['isFileNameLegal']    and not result['isFileNameExists']
        return result

    def is_file_name_legal(self, file_name):
        return not match(r'^[0-9a-zA-Z_\-\+\.]{4,64}$', file_name) is None

class DatasetProcessHandler(BaseHandler):
    def initialize(self):
        self.dataset_parser     = DatasetParser()
        self.metaset_parser     = MetasetParser()
        self.algorithms         = Algorithms()

    def post(self):
        dataset_name        = self.get_argument('datasetName')
        dataset_file_name   = self.get_argument('datasetFileName', '')
        metaset_file_name   = self.get_argument('metasetFileName', '')
        process_steps       = self.get_argument('processFlow')

        current_user        = self.get_current_user()
        dataset_file_path   = self.get_file_path(current_user, dataset_name, dataset_file_name)
        metaset_file_path   = self.get_file_path(current_user, dataset_name, metaset_file_name)
        dataset             = self.dataset_parser.get_datasets(dataset_file_path, None, None)
        metaset             = self.metaset_parser.get_metaset(metaset_file_path)

        result              = {
            'isDatasetExists': file_exists(dataset_file_path) if dataset_file_path else False,
            'isParametersLegal': self.is_parameters_legal(process_steps)
        }
        result['isSuccessful']  = True

        if result['isSuccessful']:
            try:
                process_steps       = load_json(process_steps)
                result['dataset']   = self.process_dataset(dataset, process_steps)
            except Exception as ex:
                result['isSuccessful'] = False
                logging.error('Error occurred: %s' % ex)

            result['metaset']   = metaset
        self.write(dump_json(result))

    def get_file_path(self, current_user, dataset_name, file_name):
        if not dataset_name:
            return None

        base_folder = join_path(self.application.settings['static_path'], 'uploads', current_user, dataset_name)
        return join_path(base_folder, file_name)

    def is_parameters_legal(self, process_steps):
        return True

    def process_dataset(self, dataset, process_steps):
        predicting = None
        process_steps.append({
            'algorithmName': 'tsne',
            'parameters': {}
        })

        for process_step in process_steps:
            algorithm_name = process_step['algorithmName'].replace('-', '_')
            parameters     = process_step['parameters']
            algorithm      =  self.algorithms.get_algorithm(algorithm_name)

            if algorithm:
                logging.debug('Executing algorithm %s' % algorithm_name)
                dataset    = algorithm(dataset, parameters)
                
                if dataset['predicting']['training'] or dataset['predicting']['testing']:
                    predicting = dataset['predicting']
            else:
                logging.warn('Algorithm [Name=%s] not found.' % algorithm_name)

        dataset['samples']['training'] = self.format_sample_points(dataset['samples']['training'])
        dataset['samples']['testing']  = self.format_sample_points(dataset['samples']['testing'])
        dataset['predicting']          = predicting
        return dataset

    def format_sample_points(self, dataset):
        if not dataset:
            return None
        formated_data = []
        for i in range(len(dataset)):
            formated_data.append({
                'x': dataset[i][0],
                'y': dataset[i][1]
            })

        return formated_data

class WorkbenchHandler(BaseHandler):
    def get(self):
        self.render('workbench/workbench.html')
