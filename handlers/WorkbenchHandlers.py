#!/usr/bin/python
# -*- coding: utf-8 -*-
import logging
from os import makedirs
from os.path import join as join_path
from os.path import exists as path_exists
from tornado.escape import json_encode as dump_json

from handlers.BaseHandler import BaseHandler
from utils.DatasetParser import DatasetParser
from utils.NcbiDatasetParser import NcbiDatasetParser

class DatasetUploadHandler(BaseHandler):
    def initialize(self):
        self.dataset_parser = DatasetParser()

    def post(self):
        file         = self.request.files['files[]'][0]
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
            logging.info('User [Username=%s] uploaded new dataset [%s] at %s.' % \
                (current_user, file_path, self.get_user_ip_addr()))

        self.write(dump_json(result))

    def is_file_acceptable(self, content_type, file_name, current_user, dataset_name):
        result = {
            'isContentTypeLegal': content_type == 'text/csv',
            'isDatasetNameEmpty': dataset_name == '',
            'isDatasetNameLegal': len(dataset_name) <= 64,
            'isDatasetNameExists': False,
            'isFileNameLegal': len(file_name) <= 64,
            'isFileNameExists': False,
        }
        result['isSuccessful'] = result['isContentTypeLegal'] and not result['isDatasetNameEmpty'] and \
                                 result['isDatasetNameLegal'] and not result['isDatasetNameExists'] and \
                                 result['isFileNameLegal']    and not result['isFileNameExists']
        return result

class DatasetProcessHandler(BaseHandler):
    def get(self):
        pass

class WorkbenchHandler(BaseHandler):
    def get(self):
        self.render('workbench/workbench.html')
