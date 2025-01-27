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
    task_type : str, optional
        The type of task, either 'classification' or 'regression'.
    """

    def __init__(
        self,
        input_shape,
        hidden_layers,
        output_shape,
        activation=nn.ReLU(),
        task_type="classification",
    ):
        super(FeedforwardMLP, self).__init__()
        self.activation = activation
        self.task_type = task_type

        # Define layers
        layer_dims = [input_shape] + hidden_layers + [output_shape]
        self.layers = nn.ModuleList(
            [
                nn.Linear(layer_dims[i], layer_dims[i + 1])
                for i in range(len(layer_dims) - 1)
            ]
        )

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
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)  # No activation on output layer

        if self.task_type == "classification":
            x = nn.Softmax(dim=1)(x)
        return x