"""
Module implementing a flexible feedforward multi-layer perceptron (MLP) in PyTorch.
"""

import torch.nn as nn

class FeedforwardMLP(nn.Module):
    """
    A flexible feedforward multi-layer perceptron (MLP) implemented in PyTorch.

    Parameters
    ----------
    input_dim : int
        The number of input features.
    hidden_layers : list of int
        A list containing the number of units in each hidden layer.
    output_dim : int
        The number of output units.
    activation : callable, optional
        The activation function to use between layers (default is ReLU).
    output_activation : callable, optional
        The activation function to use on the output layer. If `None` the output layer will be linear.
    dropout_rate : float, optional
        The dropout rate applied after each hidden layer (default is 0.0).
    """

    def __init__(
        self,
        input_shape,
        hidden_layers,
        output_shape,
        activation=nn.ReLU(),
        output_activation=None,
        dropout_rate=0.0,
    ):
        super(FeedforwardMLP, self).__init__()
        self.activation = activation
        self.output_activation = output_activation

        # Define the network layers
        # A dropout layer is applied after the input layer and each hidden layer
        layers = [nn.Dropout(dropout_rate)] if dropout_rate > 0.0 else []
        input_dim = input_shape

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(self.activation)
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, output_shape))
        if output_activation is not None:
            layers.append(output_activation)

        self.layers = nn.ModuleList(layers)

       

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_dim).
        """
        for layer in self.layers:
            x = layer(x)
        return x