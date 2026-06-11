"""
"""

# standard libary
import logging

# third party packages
import torch
import torch.nn as nn
import torch.optim as optim
from torcheval.metrics import MulticlassF1Score
import mlflow

# custom packages

# set up module logger
logger = logging.getLogger(__name__)


class Trainer:
    """
    A class to train a PyTorch model.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained.
    train_loader : torch.utils.data.DataLoader
        The training data loader.
    test_loader : torch.utils.data.DataLoader
        The test data loader.
    loss_fn : torch.nn.Module
        The loss function to be used.
    optimizer : torch.optim.Optimizer
        The optimizer class to be used.
    device : str
        The device to train the model on.
    task_type : str
        The type of task ('classification' or 'regression').
    test_metric : callable
        The metric to evaluate the model on the test set.

    Attributes
    ----------
    epochs_trained : int
        The number of epochs the model has been trained for.
    optimizer : torch.optim.Optimizer
        The optimizer instance used for training.
    model : torch.nn.Module
        The model to be trained.
    train_loader : torch.utils.data.DataLoader
        The training data loader.
    test_loader : torch.utils.data.DataLoader
        The test data loader.
    loss_fn : torch.nn.Module
        The loss function to be used.
    device : str
        The device to train the model on.
    task_type : str
        The type of task ('classification' or 'regression').
    test_metric : callable
        The metric to evaluate the model on the test set.
    """

    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        loss_fn,
        optimizer,
        device,
        test_metric,
        regression: bool = False,
        classification: bool = False,
    ):
        """
        Initializes the Trainer class.
        """
        assert (
            regression ^ classification
        ), "Please specify the task type: 'classification' or 'regression'."
        self._model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.device = device
        self.optimizer = optimizer
        self.regression = regression
        self.classification = classification
        self.test_metric = test_metric
        self._epochs_trained = 0

    @property
    def epochs_trained(self):
        return self._epochs_trained

    @property
    def model(self):
        return self._model

    def train_one_epoch(self):
        """
        Trains the model for one epoch.

        Returns
        -------
        float
            Training loss for the average batch.
        """
        logger.debug("Starting training one epoch.")
        self.model.train()
        loss_total = 0
        for batch_idx, (inputs, targets, path) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()

            loss_total += loss.item()
        loss_per_batch_avg = loss_total / len(self.train_loader)
        return loss_per_batch_avg

    def train(self, n_epochs) -> None:
        """
        Trains the model for a given number of epochs.

        Parameters
        ----------
        n_epochs : int
            Number of epochs to train the model for.

        Returns
        -------
        float
            Average training loss for a batch in the last epoch.
        """
        loss_per_batch_avg = 0
        for epoch in range(n_epochs):
            loss_per_batch_avg = self.train_one_epoch()
            self._epochs_trained += 1
            logger.debug(f"Epoch {self.epochs_trained} completed")

        return loss_per_batch_avg

    def get_loss_on_train_set(self) -> float:
        """
        Returns the average loss on the training set.

        Returns
        -------
        float
            Average loss on the training set.
        """
        logger.debug("Starting to calculate loss on training set.")
        logger.debug(f"{self.device}")
        self.model.eval()
        loss_total = 0
        with torch.no_grad():
            for inputs, targets, path in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss_total += loss.item()

        return loss_total / len(self.train_loader)

    def test(self) -> float:
        """
        Returns the average test score for a batch.

        Returns
        -------
        float
            Average test score for a batch.
        """
        logger.debug("Starting testing.")
        self.model.eval()
        test_score_total = 0
        with torch.no_grad():
            for inputs, targets, path in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                try:
                    test_score = self.test_metric(outputs, targets)
                    test_score_total += test_score.item()
                except TypeError:
                    test_score = self.test_metric.update(outputs, targets)

        if isinstance(self.test_metric, MulticlassF1Score):
            return self.test_metric.compute()

        return test_score_total / len(self.test_loader)
