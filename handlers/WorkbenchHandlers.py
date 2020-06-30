 #!/usr/bin/python
# -*- coding: utf-8 -*-
import logging

from concurrent.futures import ThreadPoolExecutor
from json import dump as dump_json_file
from json import load as load_json_file
from os import makedirs
from os import listdir
from os import stat as file_stat
from os.path import isfile as file_exists
from os.path import isdir as folder_exists
from os.path import join as join_path
from os.path import exists as path_exists
from tornado.concurrent import run_on_executor
from tornado.escape import json_decode as load_json
from tornado.escape import json_encode as dump_json
from tornado.gen import coroutine
from tornado.web import asynchronous
from tornado.web import HTTPError
from re import match
from re import findall

from handlers.BaseHandler import BaseHandler
from utils.Algorithms import Algorithms
from utils.DatasetParsers import DatasetParser
from utils.DatasetParsers import MetasetParser
from utils.ProjectParsers import ProjectParser

PROJECT_CONFIG_FILE_NAME = 'project-config.json'

class ProjectsSuggestionsHandler(BaseHandler):
    def initialize(self):
        self.project_parser = ProjectParser()

    @asynchronous
    def get(self):
        project_keywords     = self.get_argument('keyword', '')
        try:
            project_keywords = load_json(project_keywords)
        except:
            project_keywords = []

        current_user         = self.get_current_user()
        user_folder_path     = join_path(self.application.settings['uploads_path'], current_user)
        projects             = []
        
        if folder_exists(user_folder_path):
            projects         = self.project_parser.get_projects(user_folder_path, project_keywords)
        
        self.finish(dump_json(projects[:10]))

class DatasetUploadHandler(BaseHandler):
    def initialize(self):
        self.dataset_parser = DatasetParser()
    
    @asynchronous
    def post(self):
        file         = self.request.files['file'][0]
        file_content = file['body']
        content_type = file['content_type']
        file_name    = file['filename']
        current_user = self.get_current_user()
        project_name = self.get_argument('projectName')
        result       = self.is_file_acceptable(content_type, file_name, current_user, project_name)

        if result['isSuccessful']:
            project_folder = join_path(self.application.settings['uploads_path'], current_user, project_name)
            file_path      = join_path(project_folder, file_name)

            if not path_exists(project_folder):
                makedirs(project_folder)

            file = open(file_path, 'w')
            file.write(file_content)
            file.close()

            file_stats = self.dataset_parser.get_file_lines_columns(file_path)
            result['fileInfo']            = dict()
            result['fileInfo']['lines']   = file_stats[0]
            result['fileInfo']['columns'] = file_stats[1]
            logging.info('User [Username=%s] uploaded new file [%s] at %s.' % \
                (current_user, file_path, self.get_user_ip_addr()))

        self.finish(dump_json(result))

    def is_file_acceptable(self, content_type, file_name, current_user, project_name):
        result = {
            'isContentTypeLegal':   True,
            'isProjectNameEmpty':   project_name == '',
            'isProjectNameLegal':   self.is_file_name_legal(project_name),
            'isProjectNameExists':  False,
            'isFileNameLegal':      self.is_file_name_legal(file_name),
            'isFileNameExists':     False,
        }
        result['isSuccessful'] = result['isContentTypeLegal'] and not result['isProjectNameEmpty'] and \
                                 result['isProjectNameLegal'] and not result['isProjectNameExists'] and \
                                 result['isFileNameLegal']    and not result['isFileNameExists']
        return result

    def is_file_name_legal(self, file_name):
        return not match(r'^[0-9a-zA-Z_\-\+\.]{4,64}$', file_name) is None

class DatasetDownloadHandler(BaseHandler):
    def initialize(self):
        self.BUFFER_SIZE = 1048576  # 1 MB

    @asynchronous
    @coroutine
    def get(self):
        username     = self.get_argument('username', 'Guest')
        project_name = self.get_argument('projectName')
        file_name    = self.get_argument('fileName')
        current_user = self.get_current_user()
        file_path    = self.get_file(username, project_name, file_name)

        if not file_path:
            if not username == current_user or not username == 'Guest':
                raise HTTPError(status_code=404)

        try:
            self.set_header('Content-Type', 'application/octet-stream')
            self.set_header('Content-Length', self.get_file_size(file_path))
            self.set_header('Content-Disposition', 'attachment; filename=%s' % file_name)

            with open(file_path) as file:
                while True:
                    file_content = file.read(self.BUFFER_SIZE)
                    if not file_content:
                        break
                    self.write(file_content)
        except:
            logging.error('Error occurred while sending file stream: [File=%s]' % file_path)
        self.finish()

    def get_file(self, username, project_name, file_name):
        project_folder  = join_path(self.application.settings['uploads_path'], username, project_name)
        file_path       = join_path(project_folder, file_name)

        if not file_exists(file_path):
            return None
        return file_path

    def get_file_size(self, file_path):
        st = file_stat(file_path)
        return st.st_size

class DatasetProcessHandler(BaseHandler):
    executor = ThreadPoolExecutor(10)

    def initialize(self):
        self.dataset_parser     = DatasetParser()
        self.metaset_parser     = MetasetParser()
        self.algorithms         = Algorithms()

    @coroutine
    def post(self):
        project_name        = self.get_argument('projectName')
        dataset_file_name   = self.get_argument('datasetFileName', '')
        metaset_file_name   = self.get_argument('metasetFileName', '')
        process_steps       = self.get_argument('processFlow')

        current_user        = self.get_current_user()
        dataset_file_path   = self.get_file_path(current_user, project_name, dataset_file_name)
        metaset_file_path   = self.get_file_path(current_user, project_name, metaset_file_name)
        config_file_path    = self.get_file_path(current_user, project_name, PROJECT_CONFIG_FILE_NAME)
        dataset             = self.dataset_parser.get_datasets(dataset_file_path, None, None)
        metaset             = self.metaset_parser.get_metaset(metaset_file_path)

        result              = {
            'isDatasetExists': file_exists(dataset_file_path) if dataset_file_path else False,
            'isParametersLegal': self.is_parameters_legal(process_steps)
        }
        result['isSuccessful']  = result['isDatasetExists'] and result['isParametersLegal']

        if result['isSuccessful']:
            try:
                process_steps       = load_json(process_steps)
                result['dataset']   = yield self.process_dataset(dataset, process_steps)
                result['metaset']   = metaset
                # Save dataset and metaset of the project
                self.save_project_config(config_file_path, dataset_file_name, metaset_file_name)
            except Exception as ex:
                result['isSuccessful'] = False
                result['errorMessage'] = ex
                logging.exception(ex)

        self.finish(dump_json(result))

    def get_file_path(self, current_user, project_name, file_name):
        if not project_name:
            return None

        project_folder = join_path(self.application.settings['uploads_path'], current_user, project_name)
        return join_path(project_folder, file_name)

    def is_parameters_legal(self, process_steps):
        return True

    def save_project_config(self, config_file_path, dataset_file_name, metaset_file_name):
        project_config = {
            'datasetFileName': dataset_file_name,
            'metasetFileName': metaset_file_name
        }
        with open(config_file_path, 'w') as json_file:
            dump_json_file(project_config, json_file)

    @run_on_executor
    def process_dataset(self, dataset, process_steps):
        predicting        = None
        nearest_neighbors = None
        hierarchy         = None
        
        process_steps.append({
            'algorithmName': 'tsne',
            'parameters': {}
        })

        for process_step in process_steps:
            algorithm_name = process_step['algorithmName'].replace('-', '_')
            parameters     = process_step['parameters']
            algorithm      = self.algorithms.get_algorithm(algorithm_name)

            if algorithm:
                logging.debug('Executing algorithm %s' % algorithm_name)
                dataset    = algorithm(dataset, parameters)
                
                if dataset['predicting']['training'] or dataset['predicting']['testing']:
                    predicting = dataset['predicting']
                if dataset['nearest_neighbors']:
                    nearest_neighbors = dataset['nearest_neighbors']
                if dataset['hierarchy']:
                    hierarchy = dataset['hierarchy']
            else:
                logging.warn('Algorithm [Name=%s] not found.' % algorithm_name)

        dataset['samples']['training'] = self.format_sample_points(dataset['samples']['training'])
        dataset['samples']['testing']  = self.format_sample_points(dataset['samples']['testing'])
        dataset['predicting']          = predicting
        dataset['nearest_neighbors']   = nearest_neighbors
        dataset['hierarchy']           = hierarchy
 
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
    @asynchronous
    def get(self):
        self.render('workbench/workbench.html')
