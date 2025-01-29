import torch.nn as nn

class ModelStack(nn.Module):
    """
    MetaModel is a neural network module that sequentially applies a list of models to the input data.

    Attributes
    ----------
    models : (list)
        A list of models to be applied sequentially.
    """

    def __init__(self, models: list):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        """
        Applies the list of models sequentially to the input data.

        Parameters
        ----------
        x : torch.Tensor
            The input data tensor.

        Returns
        -------
        torch.Tensor
            The output data tensor. Output of the last model in `models`.
        """
        for model_index in range(len(self.models)):
            x = self.models[model_index](x)
        return x