"""
"""
# standard libary
import logging

# third party packages
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow

# custom packages

# set up logger
logging_level = logging.DEBUG
logger = logging.getLogger(__name__)
logger.setLevel(logging_level)
formatter = logging.Formatter("%(asctime)s %(levelname)8s - %(message)s")
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging_level)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

class Trainer:

    def __init__(self, model, train_loader, test_loader, loss_fn, optimizer, device,task_type, test_metric):
        self._model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.device = device
        self.optimizer = optimizer
        self.task_type = task_type
        self.test_metric = test_metric
        self._epochs_trained = 0

    @property
    def epochs_trained(self):
        return self._epochs_trained

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        logger.debug("enter optimizer setter")
        self._optimizer = optimizer(self.model.parameters())

    @property
    def model(self):
        return self._model
    
    @model.setter
    def model(self, model):
        self._model = model
        self._optimizer = self.optimizer(self.model.parameters())

    def train_one_epoch(self):
        logger.debug("Starting training one epoch.")
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            
            if self.task_type == 'classification':
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            elif self.task_type == 'regression':
                total += targets.size(0)
                correct += torch.sum(torch.abs(outputs - targets) < 0.5).item()  # Example threshold for regression accuracy

        if self.task_type == 'classification':
            accuracy = 100. * correct / total
        elif self.task_type == 'regression':
            accuracy = correct / total  # This is a placeholder, adjust based on your regression accuracy metric

        return train_loss / len(self.train_loader), accuracy
    
    def train(self, n_epochs):
        for epoch in range(n_epochs):
            train_loss, train_acc = self.train_one_epoch()
            self._epochs_trained += 1
            mlflow.log_metric(str(self.loss_fn)[:-2], train_loss, step=self.epochs_trained)
            print(f"Epoch {epoch} Train Loss: {train_loss:.3f} Train Acc: {train_acc:.2f}")

    def test(self):
        logger.debug("Starting testing.")
        self.model.eval()
        test_score_total = 0
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                test_score = self.test_metric(outputs, targets)
                test_score_total += test_score.item()

        return test_score_total / len(self.test_loader)
            