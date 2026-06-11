import unittest

import pandas as pd

from clearshape.pipelines.model_output_to_reporting import ModelOutputToReportingPipeline

class TestGetTrueAndPredictedValues(unittest.TestCase):

    def test_returns_correct_values_for_classifier(self):
        # Arrange
        model_output = pd.DataFrame({
            'data_type': ['images', 'images', 'trees'],
            'pred_class_id': [1, 0, 2],
            'path': ['/path/to/img1', '/path/to/img2', '/path/to/tree1'],
        })
        test_data = pd.DataFrame({
            'class_id': [0,0],
            'path': ['/path/to/cad1', '/path/to/cad2',],
        })
        data_type = "images"
        is_classifier = True

        # Act
        true_values, predicted_values = ModelOutputToReportingPipeline()._get_true_and_prediced_values(
            model_output, test_data, data_type, is_classifier
        )
        # Assert
        self.assertListEqual(true_values.to_list(), [0, 0])
        self.assertListEqual(predicted_values.to_list(), [1, 0])


class TestGetConfusionMatrix(unittest.TestCase):

    def test_returns_correct_confusion_matrix(self):
        # Arrange
        class_ids_true = pd.Series([0, 0, 1, 1, 2, 2])
        class_ids_predicted = pd.Series([0, 0, 1, 2, 1, 2])
        data_tyep = "images"
        class_ids_name_map = {0: 'cat', 1: 'dog', 2: 'tree'}

        # Act
        confusion_matrix = ModelOutputToReportingPipeline()._get_confusion_matrix(
            class_ids_true, class_ids_predicted, data_tyep, class_ids_name_map
        )
        # Assert
        expected_confusion_matrix = pd.DataFrame({
            'cat': [1., 0., 0.],
            'dog': [0., 0.5, 0.5],
            'tree': [0., 0.5, 0.5]
        }, index=['cat', 'dog', 'tree'])
        pd.testing.assert_frame_equal(confusion_matrix, expected_confusion_matrix)

        