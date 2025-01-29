import unittest

import torch
import torch.nn as nn
import dgl

from clearshape.models.treelstm import ChildSumTreeLSTMCell, RootedInTreeEncoder

# Test the ChildSumTreeLSTMCell

class TestConstructor(unittest.TestCase):
    
    def setUp(self):
        self.cell = ChildSumTreeLSTMCell(10, 5)

    def test_attributes_are_set_correctly(self):
        self.assertIsNotNone(self.cell.W_iou)
        self.assertIsNotNone(self.cell.U_iou)
        self.assertIsNotNone(self.cell.b_iou)
        self.assertIsNotNone(self.cell.W_f)
        self.assertIsNotNone(self.cell.U_f)
        self.assertIsNotNone(self.cell.b_f)

    def test_initial_weigths_are_not_zero(self):
        for param in self.cell.parameters():
            self.assertFalse(torch.equal(param.data, torch.zeros_like(param.data)))

    def test_weights_have_correct_shape(self):
        self.assertEqual(self.cell.W_iou.in_features, self.cell.input_size)
        self.assertEqual(self.cell.W_iou.out_features, 3 * self.cell.encoding_size)
        self.assertEqual(self.cell.U_iou.in_features, self.cell.encoding_size)
        self.assertEqual(self.cell.U_iou.out_features, 3 * self.cell.encoding_size)
        self.assertEqual(self.cell.b_iou.shape, (1, 3 * self.cell.encoding_size))
        self.assertEqual(self.cell.W_f.in_features, self.cell.input_size)
        self.assertEqual(self.cell.W_f.out_features, self.cell.encoding_size)
        self.assertEqual(self.cell.U_f.in_features, self.cell.encoding_size)
        self.assertEqual(self.cell.U_f.out_features, self.cell.encoding_size)
        self.assertEqual(self.cell.b_f.shape, (1, self.cell.encoding_size))

# Test the RootedInTreeEncoder
class TestRootedInTreeEncoder(unittest.TestCase):
    
    def setUp(self):
        self.encoder = RootedInTreeEncoder(10, 5)

    def test_attributes_are_set_correctly(self):
        self.assertEqual(self.encoder.input_size, 10)
        self.assertEqual(self.encoder.encoding_size, 5)
        self.assertIsNotNone(self.encoder.cell.W_iou)
        self.assertIsNotNone(self.encoder.cell.U_iou)
        self.assertIsNotNone(self.encoder.cell.b_iou)
        self.assertIsNotNone(self.encoder.cell.W_f)
        self.assertIsNotNone(self.encoder.cell.U_f)
        self.assertIsNotNone(self.encoder.cell.b_f)

    def test_initial_weigths_are_not_zero(self):
        for param in self.encoder.parameters():
            self.assertFalse(torch.equal(param.data, torch.zeros_like(param.data)))

    def test_weights_have_correct_shape(self):
        self.assertEqual(self.encoder.cell.W_iou.in_features, self.encoder.input_size)
        self.assertEqual(self.encoder.cell.W_iou.out_features, 3 * self.encoder.encoding_size)
        self.assertEqual(self.encoder.cell.U_iou.in_features, self.encoder.encoding_size)
        self.assertEqual(self.encoder.cell.U_iou.out_features, 3 * self.encoder.encoding_size)
        self.assertEqual(self.encoder.cell.b_iou.shape, (1, 3 * self.encoder.encoding_size))
        self.assertEqual(self.encoder.cell.W_f.in_features, self.encoder.input_size)
        self.assertEqual(self.encoder.cell.W_f.out_features, self.encoder.encoding_size)
        self.assertEqual(self.encoder.cell.U_f.in_features, self.encoder.encoding_size)
        self.assertEqual(self.encoder.cell.U_f.out_features, self.encoder.encoding_size)
        self.assertEqual(self.encoder.cell.b_f.shape, (1, self.encoder.encoding_size))

    def test_forward_pass_does_not_produce_nan(self):
        x = torch.randn(5, 10)
        graph = dgl.graph(([0, 1, 2], [3, 3, 4]))
        graph.ndata["node_classes"] = x
        output_h, output_c = self.encoder(graph)
        self.assertFalse(torch.isnan(output_h).any())
        self.assertFalse(torch.isnan(output_c).any())
    