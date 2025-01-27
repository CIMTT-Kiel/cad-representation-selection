import unittest
import torch
import torch.nn as nn
from clearshape.models.feedforward_mlp import FeedforwardMLP

class TestConstructor(unittest.TestCase):

    def setUp(self):
        self.mlp = FeedforwardMLP(10, [20], 5)

    def test_attributes_are_set_correctly(self):
        self.assertEqual(self.mlp.layers[0].in_features, 10)
        self.assertEqual(self.mlp.layers[0].out_features, 20)
        self.assertEqual(self.mlp.layers[-1].out_features, 5)
        self.assertIsInstance(self.mlp.activation, nn.ReLU)
        self.assertEqual(self.mlp.task_type, "classification")

    def test_initial_weigths_are_not_zero(self):
        for param in self.mlp.parameters():
            self.assertFalse(torch.equal(param.data, torch.zeros_like(param.data)))

class TestForward(unittest.TestCase):

    def setUp(self):
        self.mlp = FeedforwardMLP(10, [20], 5)

    def test_output_shape_is_5(self):
        input_tensor = torch.randn(1, 10)
        output = self.mlp(input_tensor)
        self.assertEqual(output.shape, (1, 5))

    def test_forward_pass_does_not_produce_nan(self):
        input_tensor = torch.randn(1, 10)
        output = self.mlp(input_tensor)
        self.assertFalse(torch.isnan(output).any())
    
    def test_output_shape_matches_batch_size(self):
        """Test that the forward pass works with a batch of inputs"""
        batch_size = 4
        input_tensor = torch.randn(batch_size, self.mlp.layers[0].in_features)
        output = self.mlp(input_tensor)
        self.assertEqual(output.shape, (batch_size, self.mlp.layers[-1].out_features))

if __name__ == '__main__':
    unittest.main()