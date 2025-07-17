import unittest

import pandas as pd

from clearshape.pipelines.feature_to_model_input import FeatureModelInputPipeline

pipeline = FeatureModelInputPipeline()

class TestConstructor(unittest.TestCase):
    
    def setUp(self):
        self.pipeline = pipeline

    def test_config_attribute_has_valid_entries(self):
        # train_size
        self.assertIsInstance(self.pipeline._conf.train_size, float,
            "Pipeline config train_size should be a float."
        )
        self.assertGreater(self.pipeline._conf.train_size, 0.0,
            "Pipeline config train_size should be greater than 0."
        )
        self.assertLess(self.pipeline._conf.train_size, 1.0,
            "Pipeline config train_size should be less than 1."
        )
        
        # val_size
        self.assertIsInstance(self.pipeline._conf.val_size, float,
            "Pipeline config val_size should be a float."
        )
        self.assertGreater(self.pipeline._conf.val_size, 0.0,
            "Pipeline config val_size should be greater than 0."
        )
        self.assertLess(self.pipeline._conf.val_size, 1.0,
            "Pipeline config val_size should be less than 1."
        )

        # test_size
        self.assertIsInstance(self.pipeline._conf.test_size, float,
            "Pipeline config test_size should be a float."
        )
        self.assertGreater(self.pipeline._conf.test_size, 0.0,
            "Pipeline config test_size should be greater than 0."
        )
        self.assertLess(self.pipeline._conf.test_size, 1.0,
            "Pipeline config test_size should be less than 1."
        )
        
        # balancing_factor
        self.assertIsInstance(self.pipeline._conf.balancing_factor, int,
            "Pipeline config balancing_factor should be a int."
        )

class TestGetMasterTable(unittest.TestCase):
    
    def setUp(self):
        self.pipeline = pipeline

    def test_master_table_is_dataframe(self):
        self.pipeline._get_master_table()
        self.assertIsInstance(self.pipeline._master_table, pd.DataFrame,
            "Master table should be a pandas DataFrame."
        )
    
    def test_return_is_none(self):
        result = self.pipeline._get_master_table()
        self.assertIsNone(result, "Method _get_master_table should return None.")

class TestOversample(unittest.TestCase):
    
    def setUp(self):
        self.pipeline = pipeline
        self.pipeline._master_table = pd.DataFrame({
            "path": ["path1", "path2", "path3"],
            "class_name": ["class1", "class2", "class2"],
            "class_id": [1, 2, 2],
            "volume": [100, 200, 300],
            "faces": [10, 20, 30],
            "edges": [5, 10, 15],
            "vertices": [3, 6, 9]
        })
        self.pipeline._class_size_min_required = 4

    def test_all_counts_match_class_size_min_required(self):
        self.pipeline._oversample(["class1", "class2"])
        class_sizes = self.pipeline._master_table.value_counts("class_id")
        for class_id, count in class_sizes.items():
            self.assertEqual(count, self.pipeline._class_size_min_required,
                f"Class {class_id} should have {self.pipeline._class_size_min_required} entries."
            )

class TestGetSmallClasses(unittest.TestCase):
    
    def setUp(self):
        self.pipeline = pipeline
        self.pipeline._master_table = pd.DataFrame({
            "path": ["path1", "path2", "path3"],
            "class_name": ["class1", "class2", "class2"],
            "class_id": [1, 2, 2],
            "volume": [100, 200, 300],
            "faces": [10, 20, 30],
            "edges": [5, 10, 15],
            "vertices": [3, 6, 9]
        })
        self.pipeline._class_size_min_required = 2

    def test_only_class1_is_returned(self):
        small_classes = self.pipeline._get_small_classes()
        self.assertIn("class1", small_classes)
        self.assertNotIn("class2", small_classes)

class TestCalcMinRequiredClassSize(unittest.TestCase):

    def setUp(self):
        self.pipeline = pipeline
        self.pipeline._master_table = pd.DataFrame({
            "path": ["path1", "path2", "path3"],
            "class_name": ["class1", "class2", "class2"],
            "class_id": [1, 2, 2],
            "volume": [100, 200, 300],
            "faces": [10, 20, 30],
            "edges": [5, 10, 15],
            "vertices": [3, 6, 9]
        })

    def test_min_required_class_size_is_2(self):
        min_required_size = self.pipeline._calc_min_required_class_size()
        self.assertEqual(min_required_size, 2,
            "Minimum required class size should be 2."
        )

