#!/usr/bin/python
# -*- coding: utf-8 -*-

from pandas import read_csv

class DatasetParser(object):
    def get_datasets(self, training_set_file_path, testing_set_file_path, label_column_name):
        training_set     = self.get_samples(training_set_file_path)
        testing_set      = self.get_samples(testing_set_file_path)
        training_samples = []
        training_labels  = []
        testing_samples  = []
        testing_labels   = []

        return {
            'training_samples': training_samples,
            'training_labels': training_labels,
            'testing_samples': testing_samples,
            'testing_labels': testing_labels,
        }

    def get_dataset(self, file_path):
        samples = read_csv(file_path)
        return samples.values.tolist()

    def get_file_lines_columns(self, file_path):
        number_of_lines   = 0
        number_of_columns = 0

        with open(file_path) as f:
            for number_of_lines, line in enumerate(f):
                pass
            
            number_of_columns = len(line.split(','))

        return number_of_lines, number_of_columns