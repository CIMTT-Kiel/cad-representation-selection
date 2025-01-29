# TODO implement tests
"""
Pytorch model classes implementing a TreeLSTM models for tree-structured data.
"""

__author__ = "Max Borm, Max Brede, Daniel Mansfeldt"

import dgl
import torch
import torch.nn as nn

class ChildSumTreeLSTMCell(nn.Module):
    # TODO add docstring
    def __init__(self, input_size, encoding_size):
        super(ChildSumTreeLSTMCell, self).__init__()
        self.input_size = input_size
        self.encoding_size = encoding_size
        
        self.W_iou = nn.Linear(input_size, 3 * encoding_size, bias=False)
        self.U_iou = nn.Linear(encoding_size, 3 * encoding_size, bias=False)
        self.b_iou = nn.Parameter(torch.rand(1, 3 * encoding_size))

        self.W_f = nn.Linear(input_size, encoding_size, bias=False)
        self.U_f = nn.Linear(encoding_size, encoding_size, bias=False)
        self.b_f = nn.Parameter(torch.rand(1, encoding_size))

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

class RootedInTreeEncoder(nn.Module):
    """
    Recursive neural network for tree-structured data.

    Input data must be tree that is directed from leaves to root.
    
    Parameters
    ----------
    child_sum : bool
        Whether to use the Child-Sum TreeLSTM cell. Default is True.
    n_ary : bool
        Whether to use n-ary trees. Default is False.
    input_size : int
        The size of the input features.
    encoding_size : int
        The size of the hidden state.
    num_classes : int
        The number of output classes.
    dropout : float
        The dropout rate.

    Attributes
    ----------
    input_size : int
        The size of the input features.
    encoding_size : int
        The size of the hidden state.
    dropout : float
        The dropout rate.
    cell : ChildSumTreeLSTMCell
        The Child-Sum TreeLSTM cell.

    See Also
    --------
    Paper on treeLSTMs:
    Improved semantic representations from tree-structured long short-term memory networks, Kai Sheng Tai et al., 2015
    """
    def __init__(self,input_size, encoding_size, child_sum=True, n_ary=False):
        if child_sum == n_ary:
            raise ValueError("Only one of child_sum and n_ary can be True")

        super().__init__()
        self.input_size = input_size
        self.encoding_size = encoding_size

        if child_sum:
            self.cell = ChildSumTreeLSTMCell(input_size, encoding_size)
        else:
            raise NotImplementedError("n_ary trees are not supported yet")

    def forward(self, g):
        # TODO add docstring
        device = g.device

        g.ndata["x"] = g.ndata["node_classes"]
        g.ndata["h"] = torch.zeros(g.number_of_nodes(), self.encoding_size).to(device)
        g.ndata["c"] = torch.zeros(g.number_of_nodes(), self.encoding_size).to(device)

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




