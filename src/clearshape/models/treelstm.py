"""
Pytorch model classes implementing a TreeLSTM model for tree-structured data.

Routine Listings
----------------
ChildSumTreeLSTMCell:
    A PyTorch module implementing the Child-Sum TreeLSTM cell.
STEPEncoder:
    A PyTorch module implementing the STEPEncoder.
FeedforwardMLP:
    A PyTorch module implementing a feedforward multi-layer perceptron.
MetaModel:
    A PyTorch module that sequentially applies a list of models to the input data.
"""

__author__ = "Max Borm, Max Brede, Daniel Mansfeldt"

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as nnf

class ChildSumTreeLSTMCell(nn.Module):
    def __init__(self, x_size, h_size):
        super(ChildSumTreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(torch.zeros(1, 3 * h_size))

        self.W_f = nn.Linear(x_size, h_size, bias=False)
        self.U_f = nn.Linear(h_size, h_size, bias=False)
        self.b_f = nn.Parameter(torch.zeros(1, h_size))

    def message_func(self, edges):
        return {"h": edges.src["h"], "c": edges.src["c"]}

    def reduce_func(self, nodes):
        h_tild = torch.sum(nodes.mailbox["h"], 1)
        wx = self.W_f(nodes.data["x"]).unsqueeze(1)
        uh = self.U_f(nodes.mailbox["h"])
        f = torch.sigmoid(wx + uh + self.b_f.unsqueeze(1))
        c_tild = torch.sum(f * nodes.mailbox["c"], 1)
        return {"h_tild": h_tild, "c_tild": c_tild}

    def apply_node_func(self, nodes):
        # equation (3), (5), (6)
        iou = self.W_iou(nodes.data["x"]) + self.b_iou
        if "h_tild" in nodes.data:
            iou += self.U_iou(nodes.data["h_tild"])
        i, o, u = torch.chunk(iou, 3, 1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        # equation (7)
        c = i * u
        if "c_tild" in nodes.data:
            c += nodes.data["c_tild"]
        # equation (8)
        h = o * torch.tanh(c)
        return {"h": h, "c": c}

class STEPEncoder(nn.Module):
    def __init__(self, x_size, h_size, num_classes, dropout):
        super(STEPEncoder, self).__init__()
        self.x_size = x_size
        self.h_size = h_size
        self.dropout = nn.Dropout(dropout)
        self.cell = ChildSumTreeLSTMCell(x_size, h_size)

    def forward(self, g):
        device = g.device

        g.ndata["x"] = g.ndata["node_classes"]
        g.ndata["h"] = torch.zeros(g.number_of_nodes(), self.h_size).to(device)
        g.ndata["c"] = torch.zeros(g.number_of_nodes(), self.h_size).to(device)

        dgl.prop_nodes_topo(
            g,
            self.cell.message_func,
            self.cell.reduce_func,
            apply_node_func=self.cell.apply_node_func,
        )

        root_nodes = (
            g.out_degrees() == 0
        )  # root nodes have no outgoing edges because the tree is directed leaf to root

        root_h = g.ndata["h"][root_nodes]
        return root_h




