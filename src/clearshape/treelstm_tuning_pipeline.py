"""
****************
Quick notes:
- the hyperparamerters used for each stage of the pipeline should be different
"""

# Standard Library
import logging
import yaml
import pickle
from abc import ABC, abstractmethod

# Third Party Libraries
from omegaconf import OmegaConf
import optuna
import mlflow
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import dgl
from sklearn.preprocessing import MinMaxScaler

# custom packages
from clearshape.models.feedforward_mlp import FeedforwardMLP
from clearshape.models.treelstm import RootedInTreeEncoder
from clearshape.models.modelstack import ModelStack
from clearshape.trainer import Trainer
import clearshape.constants as cons
from clearshape.dataset import FabwaveDataset

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


class TreeLSTMTuningPipeline(ABC):
    """
    # TODO add docstring
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        """
        Singleton pattern to ensure only one instance of the class is created.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_file: str):
        """
        # TODO add docstring
        """
        self._conf = OmegaConf.load(cons.PATHS.CONFIG / config_file)
        self._current_stage = None

    def _get_study(self) -> optuna.study.Study:
        """
        # TODO add docstring
        """
        match self._current_stage:
            case "train":
                return optuna.create_study(
                    direction="minimize",
                    pruner=optuna.pruners.MedianPruner(n_startup_trials=self._conf.n_jobs, n_warmup_steps=20),
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
        # TODO add docstring
        """
        with open(cons.PATHS.DATA_MODEL_INPUT / "min_max_scaler.pkl", "rb") as f:
            return pickle.load(f)

    def _load_data_sets(self, task_type: str):
        """
        Loads the training data set and the validation or test data set if the optimization stage requires it.
        """
        logger.debug(f"Loading data set for {self._current_stage} stage.")
        if task_type == "regression":
            scaler = self._load_scaler()
        # train data is the same for all stages
        self._train_data_set = FabwaveDataset(
            csv_file=cons.PATHS.DATA_MODEL_INPUT / "train.csv",
            data_type="trees",
            task_type=task_type,
            scaler=scaler,
        )
        match self._current_stage:
            case "validation":
                self._val_data_set = FabwaveDataset(
                    csv_file=cons.PATHS.DATA_MODEL_INPUT / "validation.csv",
                    data_type="trees",
                    task_type=task_type,
                    scaler=scaler,
                )
            case "test":
                self._test_data_set = FabwaveDataset(
                    csv_file=cons.PATHS.DATA_MODEL_INPUT / "test.csv",
                    data_type="trees",
                    task_type=task_type,
                    scaler=scaler,
                )

    def _set_data_loader(self, batch_size: int):
        logger.debug(f"Setting up data loader for {self._current_stage} data.")
        self._load_data_sets()
        # Train data loader is the same for each optimization stage
        self._train_data_loader = dgl.dataloading.GraphDataLoader(
            self._train_data_set, batch_size=batch_size, shuffle=True
        )
        match self._current_stage:
            case "validation":
                self._val_data_loader = dgl.dataloading.GraphDataLoader(
                    self._val_data_set, batch_size=batch_size, shuffle=True
                )
            case "test":
                self._test_data_loader = dgl.dataloading.GraphDataLoader(
                    self._test_data_set, batch_size=batch_size, shuffle=True
                )

    def _build_model(self, params: dict):
        logger.debug("Building model.")
        encoder = RootedInTreeEncoder(
            child_sum=True,
            input_size=self._conf.input_size,
            encoding_size=params["encoding_size"],
        )
        predictor = FeedforwardMLP(
            input_shape=params["encoding_size"],
            hidden_layers=params["hidden_layers"],
            output_shape=self._conf.features_total,
            task_type="regression",
        )
        model = ModelStack([encoder, predictor])
        return model

    def _get_parameters(self, trial: optuna.Trial):
        logger.debug(f"Getting parameters for trial {trial.number}")
        # get best parameters for previous stages
        match self._current_stage:
            case "train":
                layer_total = trial.suggest_int(
                    "layers_total",
                    self._conf.train.predictor.layers_total_min,
                    self._conf.train.predictor.layers_total_max,
                )
                return {
                    "batch_size": trial.suggest_int(
                        "batch_size",
                        self._conf.train.batch_size_min,
                        self._conf.train.batch_size_max,
                    ),
                    "encoding_size": trial.suggest_int(
                        "encoding_size",
                        self._conf.train.encoder.encoding_size_min,
                        self._conf.train.encoder.encoding_size_max,
                    ),
                    # layers is a list on units per layer in the predictor MLP
                    "hidden_layers": [
                        trial.suggest_int(
                            f"layer_{i}",
                            self._conf.train.predictor.units_per_layer_min,
                            self._conf.train.predictor.units_per_layer_max,
                        )
                        for i in range(layer_total)
                    ],
                }
            case "validation":
                return {}
            case "test":
                return {}

    def _load_best_parameter(self, config_file: str):
        """
        # TODO add docstring
        """
        with open(cons.PATHS.DATA_REPORTING / config_file) as f:
            return yaml.safe_load(f) or {}

    # @abstractmethod
    # def _optimize_loss_on_training_data(self, trial: optuna.Trial):
    #     pass

    # @abstractmethod
    # def _optimize_loss_on_validation_data(self, trial: optuna.Trial):
    #     """
    #     # TODO add docstring
    #     """
    #     pass

    # @abstractmethod
    # def _optimize_loss_on_test_data(self, trial: optuna.Trial):
    #     """
    #     # TODO add docstring
    #     """
    #     pass

    def _optimize_model(self):
        """
        # TODO add docstring
        """
        logger.debug("Entering optimization function.")
        study = self._get_study()
        self.best_parameter = self._load_best_parameter()
        match "train":
            case "train":
                objective = self._optimize_loss_on_training_data
            case "validation":
                objective = self._optimize_loss_on_validation_data
            case "test":
                objective = self._optimize_loss_on_test_data


        study.optimize(objective, n_trials=self._conf.n_trials, n_jobs=self._conf.n_jobs)
        # TODO complete saving best parameters to yaml file
        # save best parameters to yaml file
        best_params = study.best_params
        return best_params

    def _get_loss_function(self):
        """
        # TODO add docstring
        """
        match self._conf.loss_function:
            case "mean_squared_error":
                return nn.MSELoss()

    def _get_test_metric(self):
        """
        # TODO add docstring
        """
        match self._conf.test_metric:
            case "mean_squared_error":
                return nn.MSELoss()
    

    def _update_best_parameters(self, best_params) -> None:
        logger.debug("Updating best parameters.")
        best_parameter_path = (
            cons.PATHS.DATA_REPORTING / "tree-lstm-regressor-best-parameter.yaml"
        )
        with open(best_parameter_path, "w") as f:
            yaml.dump(best_params, f)

    @abstractmethod
    def run(self):
        pass


class TreeLSTMClassifierPipeline(TreeLSTMTuningPipeline):
    """
    # TODO add docstring
    """

    def __init__(self):
        super().__init__("classification")

    def _load_data_set(self):
        return super()._load_data_set("classification")

    def _load_best_parameters(self):
        return super()._load_best_parameters("tree-lstm-classifier-best-parameter.yaml")

    def run(self):
        pass


class TreeLSTMRegressorPipeline(TreeLSTMTuningPipeline):
    """
    # TODO add docstring
    """

    TASK_TYPE = "regression"

    def __init__(self):
        config_file = "treelstm_regressor_tuning_pipeline.yaml"
        super().__init__(config_file)

    def _load_data_sets(
        self,
    ):
        return super()._load_data_sets(self.TASK_TYPE)

    def _load_best_parameter(self):
        return super()._load_best_parameter("tree-lstm-regressor-best-parameter.yaml")

    def _optimize_loss_on_training_data(self, trial: optuna.Trial):
        """
        # TODO add docstring
        """
        with mlflow.start_run():
            # hyperparameters on model size
            parameter = self._get_parameters(trial)
            parameter.update(self.best_parameter)
            # prepare next training
            model = self._build_model(parameter)
            logger.debug(parameter["batch_size"])
            self._set_data_loader(parameter["batch_size"])
            trainer = Trainer(
                model=model,
                optimizer=optim.Adam,
                train_loader=self._train_data_loader,
                test_loader=self._train_data_loader,  # yes train data twice!
                loss_fn=self._get_loss_function(),
                test_metric=self._get_test_metric(),
                device=device,
                task_type=self.TASK_TYPE,
            )

            training_loss = trainer.test()  # initial test of model
            trial.report(training_loss, trainer.epochs_trained)
            mlflow.log_metric(str(self._conf.loss_function), training_loss, step=trainer.epochs_trained)

            # train model
            epochs_to_train_in_a_row = 10
            for epoch in range(self._conf.n_epochs // epochs_to_train_in_a_row):
                trainer.train(n_epochs=epochs_to_train_in_a_row)

                training_loss = trainer.test()
                mlflow.log_metric(str(self._conf.loss_function), training_loss, step=trainer.epochs_trained)
                trial.report(training_loss, trainer.epochs_trained)
                logger.debug(f"Epochs trained: {trainer.epochs_trained}")

                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            return training_loss

    def run(self):
        logger.info("Setting up mlflow.")
        mlflow.set_tracking_uri("http://localhost:5000")

        if "train" in self._conf.stages:
            mlflow.set_experiment("optimize_on_training_data")
            self._current_stage = "train"
            logger.info("Starting optimization on train data.")
            best_params = self._optimize_model()
            self._update_best_parameters(best_params)

        if "val" in self._conf.stages:
            # validate_model(best_params)
            pass

        if "test" in self._conf.stages:
            # test_model(best_params)
            pass
        logger.info("Pipeline completed.")


if __name__ == "__main__":
    pipeline = TreeLSTMRegressorPipeline()
    pipeline.run()
