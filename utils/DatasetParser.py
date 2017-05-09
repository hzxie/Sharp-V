#!/usr/bin/python
# -*- coding: utf-8 -*-

import logging
from pandas import read_csv

class DatasetParser(object):
    def get_datasets(self, training_set_file_path, testing_set_file_path, label_columns_name):
        training_data_frame     = self.get_dataset(training_set_file_path)
        testing_data_frame      = self.get_dataset(testing_set_file_path)
        training_data_columns   = training_data_frame.columns.values if not training_data_frame is None else None
        testing_data_columns    = testing_data_frame.columns.values if not testing_data_frame is None else None
        training_id_column_name = training_data_columns[0] if not training_data_columns is None else None
        testing_id_column_name  = testing_data_columns[0] if not testing_data_columns is None else None
        training_ids            = self.get_column_values_within_dataset(training_data_frame, training_id_column_name)
        testing_ids             = self.get_column_values_within_dataset(testing_data_frame, testing_id_column_name)
        training_labels         = self.get_column_values_within_dataset(training_data_frame, label_columns_name)
        testing_labels          = self.get_column_values_within_dataset(testing_data_frame, label_columns_name)
        training_samples        = self.get_samples(training_data_frame, [label_columns_name, training_id_column_name])
        testing_samples         = self.get_samples(testing_data_frame, [label_columns_name, testing_id_column_name])

        return {
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
            }
        }

    def get_dataset(self, file_path):
        if file_path:
            try:
                return read_csv(file_path)
            except:
                pass
        return None

    def get_column_values_within_dataset(self, data_frame, column_name):
        column_values = None
        if not data_frame is None and column_name:
            try:
                column_values = data_frame[column_name].tolist()
            except:
                logging.error('Error occurred: %s' % ex)
        return column_values

    def get_samples(self, data_frame, column_names):
        samples = None
        dummy_column_names = ['', None]

        for dcn in dummy_column_names:
            if dcn in column_names:
                column_names.remove(dcn)

        if not data_frame is None:
            try:
                df = data_frame.drop(column_names, axis=1)
                return df.values.tolist()
            except Exception as ex:
                logging.error('Error occurred: %s' % ex)
        return samples

    def get_file_lines_columns(self, file_path):
        number_of_lines   = 0
        number_of_columns = 0

        with open(file_path) as f:
            for number_of_lines, line in enumerate(f):
                pass
            
            number_of_columns = len(line.split(','))
        return number_of_lines, number_of_columns
