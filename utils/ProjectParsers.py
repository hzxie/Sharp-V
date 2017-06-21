#!/usr/bin/python
# -*- coding: utf-8 -*-

import logging
from io import StringIO
from json import dump as dump_json_file
from json import load as load_json_file
from os import makedirs
from os import listdir
from os import stat as file_stat
from os.path import isfile as file_exists
from os.path import isdir as folder_exists
from os.path import join as join_path
from os.path import exists as path_exists
from sets import Set

PROJECT_CONFIG_FILE_NAME = 'project-config.json'

class ProjectParser(object):
    def get_projects(self, user_folder_path, project_keywords = ['']):
        projects             = []
        project_names        = Set()
        for project_name in listdir(user_folder_path):
            if folder_exists(join_path(user_folder_path, project_name)):
                for keyword in project_keywords:
                    if (project_name.lower().find(keyword.lower()) != -1) and not project_name in project_names:
                        project_folder_path = join_path(user_folder_path, project_name) 
                        project_files       = self.get_project_files(project_folder_path)                       
                        project_names.add(project_name)
                        projects.append({
                            'projectName': project_name,
                            'projectFiles': project_files
                        })
        return projects

    def get_project_files(self, project_folder_path):
        project_files      = [file for file in listdir(project_folder_path)]
        dataset_file_name  = None
        metaset_file_name  = None
        json_file_name     = PROJECT_CONFIG_FILE_NAME
        project_config     = None

        if json_file_name in project_files:
            project_files.remove(json_file_name)
            with open(join_path(project_folder_path, json_file_name)) as json_file:
                project_config    = load_json_file(json_file)
                dataset_file_name = project_config['datasetFileName']
                metaset_file_name = project_config['metasetFileName']

        return {
            'candidateFiles': project_files,
            'datasetName': dataset_file_name,
            'metasetName': metaset_file_name
        }
