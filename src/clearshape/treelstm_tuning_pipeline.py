"""
A class to optimize hyperparameters for a Tree-LSTM model using Optuna.

Examples
--------
Start MLFlow tracking via `mlflow ui` then run the pipeline:

>>> pipeline = TreeLSTMTuningPipeline("treelstm_classifier_tuning_pipeline.yaml", classification=True)
>>> pipeline.run()
"""

# Standard Library
import logging
import pickle
import yaml

# Third Party Libraries
import dgl
import mlflow
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from torcheval.metrics import MulticlassF1Score

# custom packages
import clearshape.constants as cons
from clearshape.dataset import FabwaveDataset
from clearshape.models.feedforward_mlp import FeedforwardMLP
from clearshape.models.modelstack import ModelStack
from clearshape.models.treelstm import RootedInTreeEncoder
from clearshape.trainer import Trainer

# set up logger
logging_level = logging.DEBUG
logger = logging.getLogger(__name__)
logger.setLevel(logging_level)
formatter = logging.Formatter("%(asctime)s %(levelname)8s - %(message)s")
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging_level)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")


class TreeLSTMTuningPipeline():
    """
    A class to optimize hyperparameters for a Tree-LSTM model using Optuna.

    Parameters
    ----------
    config_file : str
        The name of the configuration file.
    regression : bool
        Whether the task is a regression task.
    classification : bool
        Whether the task is a classification task.
    
    Attributes
    ----------
    _conf : OmegaConf
        The configuration object.
    _current_stage : str    
        The current stage of the optimization pipeline.
    best_model : dict   
        A dictionary containing the best model and its loss.
    regression : bool
        Whether the task is a regression task.
    classification : bool   
        Whether the task is a classification
    """


    def __init__(self, config_file: str, regression:bool=False, classification: bool=False):
        """
        Initializes the TreeLSTMTuningPipeline class.
        """
        self._conf = OmegaConf.load(cons.PATHS.CONFIG / config_file)
        # TODO verify that the configuration file has valid entries
        self._current_stage = None
        self.best_model = {"model": None, "test_score": float("inf")}
        assert regression ^ classification, "Choose either regression or classification task"
        self.regression = regression
        self.classification = classification

    def _get_study(self) -> optuna.study.Study:
        """
        Get an Optuna study object based on the current stage of the optimization pipeline.

        Returns
        -------
        optuna.study.Study
            An Optuna study object.
        """
        match self._current_stage:
            case "train":
                return optuna.create_study(
                    direction="minimize",
                    pruner=optuna.pruners.MedianPruner(
                        n_startup_trials=self._conf.n_jobs, n_warmup_steps=20
                    ),
                )
            case "validation":
                return optuna.create_study(
                    direction="minimize",
                )
            case "test":
                return optuna.create_study(
                    direction="minimize",
                )

    def _load_scaler(self):
        """
        Load the MinMaxScaler object from a pickle file.

        Returns
        -------
        MinMaxScaler
            The MinMaxScaler object.
        """
        with open(cons.PATHS.DATA_MODEL_INPUT / "min_max_scaler.pkl", "rb") as f:
            return pickle.load(f)

    def _load_data_sets(self):
        """
        Loads the training data set and the validation or test data set if the optimization stage requires it.

        The data sets are set as attributes of the class.

        Returns
        -------
        None
        """
        logger.debug(f"Loading data set for {self._current_stage} stage.")
        scaler = self._load_scaler() if self.regression else None
        # train data is the same for all stages
        self._train_data_set = FabwaveDataset(
            csv_file=cons.PATHS.DATA_MODEL_INPUT / "train.csv",
            data_type="trees",
            regression=self.regression,
            classification=self.classification,
            scaler=scaler,
        )
        match self._current_stage:
            case "validation":
                self._val_data_set = FabwaveDataset(
                    csv_file=cons.PATHS.DATA_MODEL_INPUT / "validation.csv",
                    data_type="trees",
                    regression=self.regression,
                    classification=self.classification,
                    scaler=scaler,
                )
            case "test":
                self._test_data_set = FabwaveDataset(
                    csv_file=cons.PATHS.DATA_MODEL_INPUT / "test.csv",
                    data_type="trees",
                    regression=self.regression,
                    classification=self.classification,
                    scaler=scaler,
                )

    def _set_data_loader(self, batch_size: int):
        """
        Set the data loader for the current stage of the optimization pipeline.
        
        Parameters
        ----------
        batch_size : int
            The batch size for the data loader.

        Returns
        -------
        None
        """
        logger.debug(f"Setting up data loader for {self._current_stage} data.")
        self._load_data_sets()
        # Train data loader is the same for each optimization stage
        self._train_data_loader = dgl.dataloading.GraphDataLoader(
            self._train_data_set, batch_size=batch_size, shuffle=True
        )
        match self._current_stage:
            case "train":
                self._test_data_loader = dgl.dataloading.GraphDataLoader(
                    self._train_data_set, batch_size=batch_size, shuffle=True
                )
            case "validation":
                self._test_data_loader = dgl.dataloading.GraphDataLoader(
                    self._val_data_set, batch_size=batch_size, shuffle=True
                )
            case "test":
                self._test_data_loader = dgl.dataloading.GraphDataLoader(
                    self._test_data_set, batch_size=batch_size, shuffle=True
                )

    def _build_model(self, params: dict):
        """
        Build a model based on the parameters provided.

        Parameters
        ----------
        params : dict
            A dictionary containing the parameters to build the model.
        
        Returns
        -------
        ModelStack
            The built model.
        """
        logger.debug("Building model.")
        encoder = RootedInTreeEncoder(
            child_sum=True,
            input_size=self._conf.input_size,
            encoding_size=params["encoding_size"],
        )
        predictor = FeedforwardMLP(
            input_shape=params["encoding_size"],
            hidden_layers=params["hidden_layers"],
            output_shape=self._conf.output_shape,
            task_type="regression",
        )
        model = ModelStack([encoder, predictor])
        return model

    def _get_parameters_to_tune(self, trial: optuna.Trial):
        """
        Get the hyperparameters to tune for the current stage of the optimization pipeline.
        
        Parameters
        ----------
        trial : optuna.Trial
            The Optuna trial object for which parameters are being retrieved.
        
        Returns
        -------
        dict
            A dictionary containing the hyperparameters to tune.
        """
        match self._current_stage:
            case "train":
                layers_total = trial.suggest_int(
                    "layers_total",
                    self._conf.train.predictor.layers_total_min,
                    self._conf.train.predictor.layers_total_max,
                )
                return {
                    "encoding_size": (
                        trial.suggest_int(
                            "encoding_size",
                            self._conf.train.encoder.encoding_size_min,
                            self._conf.train.encoder.encoding_size_max,
                            step=self._conf.train.encoder.encoding_size_step,
                        )
                    ),
                    "hidden_layers": [
                        trial.suggest_int(
                            f"layer_{i}",
                            self._conf.train.predictor.units_per_layer_min,
                            self._conf.train.predictor.units_per_layer_max,
                        )
                        for i in range(layers_total)
                    ],
                }
            case "validation":
                return {
                    "learning_rate": trial.suggest_float(
                        "learning_rate",
                        self._conf.validation.learning_rate_min,
                        self._conf.validation.learning_rate_max,
                    ),
                    "dropout_rate": trial.suggest_float(
                        "dropout_rate",
                        self._conf.validation.encoder.dropout_rate_min,
                        self._conf.validation.encoder.dropout_rate_max,
                    ),
                }
            case "test":
                return {
                    "batch_size": trial.suggest_int(
                        "batch_size",
                        self._conf.test.batch_size_min,
                        self._conf.test.batch_size_max,
                        step=self._conf.test.batch_size_step,
                    )
                }

    def _get_fixed_parameters(self):
        """
        Get the fixed hyperparameters for the current stage of the optimization pipeline.

        Returns
        -------
        dict
            A dictionary containing the fixed hyperparameters.
        """
        match self._current_stage:
            case "train":
                return {
                    "batch_size": self._conf.train.batch_size,
                    "optimizer": self._conf.train.optimizer,
                }
            case "validation":
                return {
                    "batch_size": self._conf.validation.batch_size,
                    "optimizer": self._conf.train.optimizer,
                }
            case "test":
                return {
                    "optimizer": self._conf.train.optimizer,
                }

    def _get_parameters(self, trial: optuna.Trial) -> dict:
        """
        Get parameters for an Optuna trial.

        This method retrieves a set of parameters consisting of:
        - Hyperparameters to tune in the current stage.
        - Tuned hyperparameters from previous stages.
        - Fixed hyperparameters for the current stage.

        It ensures that there are no overlapping keys among the sets of parameters.

        Parameters
        ----------
        trial : optuna.Trial
            The Optuna trial object for which parameters are being retrieved.

        Returns
        -------
        dict
            A dictionary containing the combined set of parameters for the trial.
        """
        logger.debug(f"Getting parameters for trial {trial.number}")
        parameters_for_trial = {}
        # get best parameters for previous stages
        parameters_to_tune = self._get_parameters_to_tune(trial)
        fixed_parameters = self._get_fixed_parameters()
        best_parameters = self._load_best_parameter()

        # assert the set of parameters do not overlap
        assert not set(parameters_to_tune.keys()).intersection(
            set(fixed_parameters.keys())
        )
        assert not set(parameters_to_tune.keys()).intersection(
            set(best_parameters.keys())
        )
        assert not set(fixed_parameters.keys()).intersection(
            set(best_parameters.keys())
        )

        parameters_for_trial.update(best_parameters)
        parameters_for_trial.update(parameters_to_tune)
        parameters_for_trial.update(fixed_parameters)
        return parameters_for_trial

    def _load_best_parameter(self):
        """
        Load the best parameter configuration from a YAML file.

        Returns
        -------
        dict
            A dictionary containing the best parameter configuration.
        """
        with open(cons.PATHS.DATA_REPORTING / self._conf.file_best_parameter) as f:
            return yaml.safe_load(f) or {}

    def _get_optimizer(self, optimizer: str):
        """
        Get the optimizer for the model.

        Parameters
        ----------
        optimizer : str
            The name of the optimizer.

        Returns
        -------
        torch.optim.Optimizer
            The optimizer instance.

        Raises
        ------
        ValueError
            If the optimizer is not recognized.

        Returns
        -------
        torch.optim.Optimizer
            The optimizer instance.
        """
        match optimizer:
            case "adam":
                return optim.Adam
            case "sgd":
                return optim.SGD
            case "rmsprop":
                return optim.RMSprop

    def _optimize_loss_on_training_data(self, trial: optuna.Trial):
        """
        Optimize the loss on the training data.

        Parameters
        ----------
        trial : optuna.Trial
            The Optuna trial object.
        
        Returns
        -------
        float
            The loss on the training data.
        """
        with mlflow.start_run():
            parameter = self._get_parameters(trial)
            logger.debug(parameter)
            mlflow.log_params(parameter)

            model, trainer = self._initialize_model_and_trainer(parameter)

            self._test_and_log(trainer, trial)

            self._train_model(trainer, trial)

            loss = self._test_and_log(trainer, trial)

            return loss

    def _optimize_loss_on_validation_data(self, trial: optuna.Trial):
        """
        Optimize the loss on the validation data.

        Parameters
        ----------
        trial : optuna.Trial
            The Optuna trial object.
        
        Returns
        -------
        float
            The loss on the validation data.
        """
        with mlflow.start_run():
            parameter = self._get_parameters(trial)
            logger.debug(parameter)
            mlflow.log_params(parameter)

            model, trainer = self._initialize_model_and_trainer(parameter)

            self._test_and_log(trainer, trial)

            self._train_model(trainer, trial)

            loss = self._test_and_log(trainer, trial)

            return loss

    # TODO rename this function. The model is not nessarily optimized on training loss.
    def _optimize_loss_on_test_data(self, trial: optuna.Trial):
        """
        Optimize the loss on the test data.

        Parameters
        ----------
        trial : optuna.Trial
            The Optuna trial object.

        Returns
        -------
        float
            The loss on the test data.
        """
        with mlflow.start_run():
            parameter = self._get_parameters(trial)
            logger.debug(parameter)
            mlflow.log_params(parameter)

            model, trainer = self._initialize_model_and_trainer(parameter)

            # calculate training loss and test score before training
            self._assess_initial_model(trainer, trial)

            test_score = self._train_model(trainer, trial)

            if test_score < self.best_model["test_score"]:
                self.best_model["model"] = model
                self.best_model["test_score"] = test_score

            return test_score
        
    def _assess_initial_model(self, trainer: Trainer, trial: optuna.Trial):
        """
        Assess the initial model before training.

        Parameters
        ----------
        trainer : Trainer
            The Trainer object to assess the model.

        Returns
        -------
        float
            The loss on the test data.
        """
        training_loss = trainer.get_loss_on_train_set()
        test_score = trainer.test()
        mlflow.log_metric(
            f"{str(self._conf.loss_function)} on training data",
            training_loss,
            step=trainer.epochs_trained,
        )
        mlflow.log_metric(
            f"{str(self._conf.test_metric)} on test data of stage",
            test_score,
            step=trainer.epochs_trained,
        )
        trial.report(training_loss, trainer.epochs_trained)
        trial.report(test_score, trainer.epochs_trained)

    def _train_model(self, trainer: Trainer, trial: optuna.Trial):
        """
        Train the model for a number of epochs.

        Parameters
        ----------
        trainer : Trainer
            The Trainer object to train the model. Also defines the number of epochs to train.

        Returns
        -------
        None
        """
        epochs_to_train_in_a_row = 10
        for _ in range(self._conf.n_epochs // epochs_to_train_in_a_row):
            training_loss = trainer.train(n_epochs=epochs_to_train_in_a_row)
            test_score = trainer.test()
            logger.debug(f"Test score: {test_score}")
            # log training loss
            mlflow.log_metric(
                f"{str(self._conf.loss_function)} on training data",
                training_loss,
                step=trainer.epochs_trained,
            )
            # log test loss
            mlflow.log_metric(
                f"{str(self._conf.test_metric)} on test data of stage",
                test_score,
                step=trainer.epochs_trained,
            )

            logger.debug("Reporting training loss and test score to Optuna.")
            trial.report(training_loss, trainer.epochs_trained)
            trial.report(test_score, trainer.epochs_trained)
            logger.info(f"Epochs trained: {trainer.epochs_trained}")

            if trial.should_prune():
                raise optuna.TrialPruned()
        return test_score


    def _initialize_model_and_trainer(self, parameter: dict):
        """
        Initialize the model and the trainer with the given parameters.

        Parameters
        ----------
        parameter : dict
            A dictionary containing the parameters to initialize the model and trainer.

        Returns
        -------
        ModelStack, Trainer
            The initialized model and trainer. 
        """
        model = self._build_model(parameter)
        self._set_data_loader(parameter["batch_size"])
        trainer = Trainer(
            model=model,
            optimizer=self._get_optimizer(parameter["optimizer"]),
            train_loader=self._train_data_loader,
            test_loader=self._test_data_loader,
            loss_fn=self._get_loss_function(),
            test_metric=self._get_test_metric(),
            device=device,
            regression=self.regression,
            classification=self.classification,
        )
        return model, trainer

    def _optimize_model(self):
        """
        Optimize the model based on the current stage of the optimization pipeline.

        Returns
        -------
        dict
            A dictionary containing the best tuned parameters.

        Raises
        ------
        ValueError
            If the current stage is not recognized.
        """
        logger.debug("Entering optimization function.")
        study = self._get_study()
        self.best_parameter = self._load_best_parameter()
        # set objective function based on current stage
        match self._current_stage:
            case "train":
                objective = self._optimize_loss_on_training_data
            case "validation":
                objective = self._optimize_loss_on_validation_data
            case "test":
                objective = self._optimize_loss_on_test_data

        study.optimize(
            objective, n_trials=self._conf.n_trials, n_jobs=self._conf.n_jobs
        )
        best_params = study.best_params
        return best_params

    # TODO refactor get_loss_function and get_test_metric. WET code!
    def _get_loss_function(self):
        """
        Instantiate the loss function based on the configuration.

        Returns
        -------
        callable
            The loss function.

        Raises
        ------
        ValueError
            If the loss function is not recognized.
        """
        match self._conf.loss_function:
            case "mean_squared_error":
                return nn.MSELoss()
            case "cross_entropy":
                return nn.CrossEntropyLoss()
            case "f1":
                return MulticlassF1Score(num_classes=self._conf.output_shape)
            case _ :
                raise ValueError(f"Loss function {self._conf.loss_function} not recognized.")

    def _get_test_metric(self):
        """
        Instantiate the test metric based on the configuration.

        Returns
        -------
        callable
            The test metric.
        
        Raises
        ------
        ValueError
            If the test metric is not recognized.
        """
        match self._conf.test_metric:
            case "mean_squared_error":
                return nn.MSELoss()
            case "cross_entropy":
                return nn.CrossEntropyLoss()
            case "f1":
                return MulticlassF1Score(num_classes=self._conf.output_shape)
            case _:
                raise ValueError(f"Test metric {self._conf.test_metric} not recognized.")

    def _save_best_tuned_parameters(self, best_tuned_parameter: dict):
        """
        Save the best tuned parameters to a YAML file.

        Parameters
        ----------
        best_params : dict
            A dictionary containing the best tuned parameters. Returned by optuna.

        Returns
        -------
        None
        """
        logger.debug("Saving best tuned parameters.")
        try:
            # put layer sizes in a list
            best_tuned_parameter["hidden_layers"] = [
                best_tuned_parameter[f"layer_{i}"]
                for i in range(best_tuned_parameter["layers_total"])
            ]
            # remove individual layer sizes
            for i in range(best_tuned_parameter["layers_total"]):
                del best_tuned_parameter[f"layer_{i}"]

            # remove redundant entries
            del best_tuned_parameter["layers_total"]
        except KeyError:
            logger.info(
                f"No hidden layers to save for this stage. Current stage: {self._current_stage}"
            )

        best_parameter = self._load_best_parameter()
        best_parameter.update(best_tuned_parameter)
        logger.debug(f"best parameter before saving: {best_parameter}")
        best_parameter_path = cons.PATHS.DATA_REPORTING / self._conf.file_best_parameter
        with open(best_parameter_path, "w") as f:
            yaml.dump(best_parameter, f)

    def run(self):
        """
        Run the optimization pipeline.

        Returns
        -------
        None
        """
        logger.info(f"Starting pipeline. Tuning {'classifier' if self.classification else 'regressor'} model.")
        logger.info("Setting up mlflow. Tracking URI: http://localhost:5000")
        mlflow.set_tracking_uri("http://localhost:5000")

        if "train" in self._conf.stages:
            logger.info("Starting optimization on train data.")
            mlflow.set_experiment("optimize_on_training_data")
            self._current_stage = "train"
            best_params = self._optimize_model()

            self._save_best_tuned_parameters(best_params)

        if "validation" in self._conf.stages:
            logger.info("Starting optimization on validation data.")
            mlflow.set_experiment("optimize_on_validation_data")
            self._current_stage = "validation"
            best_params = self._optimize_model()

            self._save_best_tuned_parameters(best_params)

        if "test" in self._conf.stages:
            logger.info("Starting optimization on test data.")
            mlflow.set_experiment("optimize_on_test_data")
            self._current_stage = "test"
            best_params = self._optimize_model()

            self._save_best_tuned_parameters(best_params)

            # save best model (classifier or regressor)
            model_type = "classifier" if self.classification else "regressor"

            assert self.best_model["model"].state_dict() is not None, "No model to save."
            
            torch.save(self.best_model["model"].state_dict(), cons.PATHS.DATA_MODELS / f"trees-{model_type}.pth")

        logger.info("Pipeline completed.")


    


if __name__ == "__main__":
    #pipeline = TreeLSTMTuningPipeline("treelstm_classifier_tuning_pipeline.yaml", classification=True)
    pipeline = TreeLSTMTuningPipeline("treelstm_regressor_tuning_pipeline.yaml", regression=True)
    pipeline.run()
