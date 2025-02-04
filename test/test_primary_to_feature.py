import unittest
import dgl

from clearshape.primary_to_feature import PrimaryFeaturePipeline
import clearshape.constants as cons

class TestNew(unittest.TestCase):

    def test_same_instance_is_returned_after_second_call(self):
        instance1 = PrimaryFeaturePipeline()
        instance2 = PrimaryFeaturePipeline()
        self.assertIs(instance1, instance2)

class TestInit(unittest.TestCase):

    def setUp(self):
        self.pipeline = PrimaryFeaturePipeline()

    def test_conf_is_loaded_correctly(self):
        self.assertIsNotNone(self.pipeline._conf)

    def test_step_path_generator_is_initialized_correctly(self):
        self.assertIsNotNone(self.pipeline._step_path_generator)

    def test_targets_is_initialized_correctly(self):
        self.assertEqual(self.pipeline._targets, [])

    def test_known_classes_is_initialized_correctly(self):
        self.assertEqual(self.pipeline._known_classes, [])

class TestGetNextStepPath(unittest.TestCase):

    def setUp(self):
        self.pipeline = PrimaryFeaturePipeline()

    def test_file_to_process_is_updated_correctly(self):
        self.pipeline._get_next_step_path()
        self.assertIsNotNone(self.pipeline._file_to_process)

class TestConvertToTree(unittest.TestCase):

    def setUp(self):
        self.pipeline = PrimaryFeaturePipeline()

    def test_step_tree_attribute_is_dgl_graph(self):
        self.pipeline._get_next_step_path()
        self.pipeline._convert_to_tree()
        self.assertIsInstance(self.pipeline._step_tree, dgl.DGLGraph)

class TestSaveData(unittest.TestCase):

    def setUp(self):
        self.pipeline = PrimaryFeaturePipeline()

    def test_data_is_saved_correctly(self):
        self.pipeline._get_next_step_path()
        self.pipeline._convert_to_tree()
        self.pipeline._save_data()
        self.assertTrue((cons.PATHS.DATA_FEATURE / "trees/fabwave/Pipe_Fittings/44965k431.bin").exists())

    def test_file_is_not_empty(self):
        self.pipeline._get_next_step_path()
        self.pipeline._convert_to_tree()
        self.pipeline._save_data()
        self.assertGreater((cons.PATHS.DATA_FEATURE / "trees/fabwave/Pipe_Fittings/44965k431.bin").stat().st_size, 0)

if __name__ == '__main__':
    unittest.main()