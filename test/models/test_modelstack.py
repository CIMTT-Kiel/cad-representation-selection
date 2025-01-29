import unittest
import torch
import torch.nn as nn
from clearshape.models.modelstack import ModelStack

class TestConstructor(unittest.TestCase):

    def setUp(self):
        model_1 = nn.Linear(10, 5)
        model_2 = nn.Linear(5, 1)
        self.modelstack = ModelStack([model_1, model_2])

    def test_attributes_are_set_correctly(self):
        self.assertEqual(len(self.modelstack.models), 2)
        self.assertIsInstance(self.modelstack.models[0], nn.Linear)
        self.assertIsInstance(self.modelstack.models[1], nn.Linear)

    def test_model_weigths_are_accessible(self):
        for model in self.modelstack.models:
            self.assertIsNotNone(model.weight)
            self.assertIsNotNone(model.bias)

class TestForward(unittest.TestCase):

    def setUp(self):
        model_1 = nn.Linear(10, 5)
        model_2 = nn.Linear(5, 1)
        self.modelstack = ModelStack([model_1, model_2])

    def test_output_shape_is_1(self):
        input_tensor = torch.randn(1, 10)
        output = self.modelstack(input_tensor)
        self.assertEqual(output.shape, (1, 1))

    def test_forward_pass_does_not_produce_nan(self):
        input_tensor = torch.randn(1, 10)
        output = self.modelstack(input_tensor)
        self.assertFalse(torch.isnan(output).any())
    
    def test_output_shape_matches_batch_size(self):
        """Test that the forward pass works with a batch of inputs"""
        batch_size = 4
        input_tensor = torch.randn(batch_size, 10)
        output = self.modelstack(input_tensor)
        self.assertEqual(output.shape, (batch_size, 1))