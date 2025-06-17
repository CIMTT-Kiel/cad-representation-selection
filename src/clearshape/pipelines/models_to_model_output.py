# %%

# TODO: add all docstring

"""
This module contains the class `ModelsModelOutputPipeline` which is responsible
for loading the models from the `data/models` directory and outputting their
predictions on the test data set to the `data/model_output` directory.

Notes
----- The pipeline assumes that files containing the models follow this naming
convention:

    `{images, trees or invariants}-{regressor or classifier}.pth`
"""

# Standard Library
import logging
import pickle
import dgl.data
import yaml

# Third Party Libraries
import dgl
import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

# custom packages
import clearshape.constants as cons
from clearshape.dataset import FabwaveDataset
from clearshape.models.feedforward_mlp import FeedforwardMLP
from clearshape.models.modelstack import ModelStack
from clearshape.models.treelstm import RootedInTreeEncoder

# set up logger
logging_level = logging.DEBUG
logger = logging.getLogger(__name__)
logger.setLevel(logging_level)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)8s - %(message)s")
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging_level)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelsModelOutputPipeline:
    """
    Pipeline to load models from the `data/6_models` directory and output
    their predictions on the test data set to the `data/model_output` directory.

    Classifiers and regressors are handled separately. The predictions of all classifiers and regressors are put in a single output file respectively.

    Output files:
    - `data/model_output/classifiers_output.csv`
    - `data/model_output/regressors_output.csv`

    Parameters
    ----------
    None

    Attributes
    ----------
    model_path_generator : generator
        Generator object to iterate over the files in the `data/models` directory.
    models : dict
        Dictionary containing the paths to the models.
    model : torch.nn.Module
        Current model being used in the pipeline.
    data_loader : torch.utils.data.DataLoader
        DataLoader object for the test data set.
    model_path : pathlib.Path
        Path to the current model being used in the pipeline.
    data_type : str
        Type of data the current model is trained on. (images, trees or invariants)
    model_type : str
        Type of model being used. (regressor or classifier)

    """

    def __init__(self):
        self.model_path_generator = cons.PATHS.DATA_MODELS.iterdir()
        self.models = self._set_models()

    def _get_models_by_type(self, model_type: str):
        """
        Get a dict of all models of the specified type (classifier or regressor)
        in the `data/models` directory.

        Parameters
        ----------
        model_type : str
            Type of models to retrieve. Must be either "classifier" or "regressor".

        Returns
        -------
        models : dict
            Dictionary containing the paths to the models of the specified type.
            The keys are the models file stem and the values are the paths to the models.
        """
        if model_type not in {"classifier", "regressor"}:
            raise ValueError("model_type must be either 'classifier' or 'regressor'")

        models = {}
        for path in cons.PATHS.DATA_MODELS.iterdir():
            if path.stem.endswith(model_type):
                models[path.stem] = {"path": path}

        return models

    def _set_models(self):
        """
        Sets the models attribute to a dictionary containing the paths to the models.
        """
        self.models = {
            path.stem: {"path": path} for path in cons.PATHS.DATA_MODELS.iterdir()
        }

    def _initialize_tree_lstm(self, task_type) -> torch.nn.Module:
        """
        Initialize the TreeLSTM model using the best parameters found during the hyperparameter search.

        Returns
        -------
        model : torch.nn.Module
            The TreeLSTM model with randomly initialized weights.
        """
        logger.info(f"Initializing TreeLSTM model")

        model_parameter = OmegaConf.load(
            cons.PATHS.DATA_MODELS / f"tree-lstm-{task_type}-best-parameter.yaml"
        )
        tuning_pipeline_config = OmegaConf.load(
            cons.PATHS.CONFIG / f"treelstm_{task_type}_tuning_pipeline.yaml"
        )

        encoder = RootedInTreeEncoder(
            input_size=tuning_pipeline_config.input_size,
            encoding_size=model_parameter.encoding_size,
            child_sum=True,
        )
        predictor = FeedforwardMLP(
            input_shape=model_parameter.encoding_size,
            hidden_layers=model_parameter.hidden_layers,
            output_shape=tuning_pipeline_config.output_shape,
        )
        model = ModelStack([encoder, predictor])
        return model

    def _initialize_image_model(
        self,
    ) -> torch.nn.Module:
        pass

    def _initialize_invariant_model(
        self,
    ) -> torch.nn.Module:
        pass

    def _load_model(self, path) -> None:
        """
        Load the models from the `data/models` directory.

        Parameters
        ----------
        path : pathlib.Path
            Path to the models state dict.

        Returns
        -------
        None
        """
        logger.info(f"Loading model from {path}")
        model = None
        model_type = path.stem.split("-")[-1]  # regressor or classifier
        data_type = path.stem.split("-")[0]  # images, trees or invariants
        match (data_type, model_type):
            case ("trees", "classifier"):
                model = self._initialize_tree_lstm("classifier")
            case ("trees", "regressor"):
                model = self._initialize_tree_lstm("regressor")
            case ("images", "classifier"):
                model = self._initialize_image_model("classifier")
            case ("images", "regressor"):
                model = self._initialize_image_model("regressor")
            case ("invariants", "classifier"):
                model = self._initialize_invariant_model("classifier")
            case ("invariants", "regressor"):
                model = self._initialize_invariant_model("regressor")
        if model is None:
            raise ValueError(
                f"Invalid model type: {data_type}-{model_type}. Supported types are: images, trees, invariants for data and regressor, classifier for model type."
            )

        model.load_state_dict(
            torch.load(path, weights_only=True, map_location=device), strict=False
        )
        model.to(device)
        model.eval()
        return model

    def _get_data_loader(self, task_type, data_type, scaler=None) -> None:
        """
        Set the DataLoader object for the test data set.

        Returns
        -------
        None
        """

        data_set = FabwaveDataset(
            cons.PATHS.DATA_MODEL_INPUT / "test.csv",
            data_type=data_type,
            regression=task_type == "regression",
            classification=task_type == "classification",
            scaler=scaler,
        )
        match data_type:
            case "images":
                data_loader = DataLoader(data_set, batch_size=32, shuffle=False)
            case "invariants":
                data_loader = DataLoader(data_set, batch_size=32, shuffle=False)
            case "trees":
                data_loader = dgl.dataloading.GraphDataLoader(
                    data_set, batch_size=32, shuffle=False
                )
        return data_loader

    def _compute_predictions(self) -> pd.DataFrame:
        """
        Compute the predictions of the model on the test data set.

        Returns
        -------
        predictions : pd.DataFrame
            DataFrame containing the predictions of the model on the test data
            set. Depending on the model type, the DataFrame will have either one
            column for the predicted class id or four columns for the predicted
            volume, faces, edges, and vertices.
        """
        match self.model_type:
            case "classifier":
                assert predictions.shape[1] == 1
            case "regressor":
                columns = self.data_loader.dataset.data.columns[-4:]
                predictions = torch.cat(predictions, dim=0).cpu().detach().numpy()
                # scale predictions
                self._load_scaler()
                predictions = self.scaler.inverse_transform(predictions)
                predictions = pd.DataFrame(
                    predictions, columns=[f"pred_{column}" for column in columns]
                )
                assert predictions.shape[1] == 4

        return predictions

    def _get_scaler(self, path):
        with open(path, "rb") as scaler_file:
            return pickle.load(scaler_file)

    def _get_targets(self) -> pd.DataFrame:
        """
        Get the targets of the test data set.

        Returns
        -------
        targets : pd.DataFrame
            DataFrame containing the targets of the test data set. Depending on
            the model type, the DataFrame will have either one column for the
            class id or four columns for the volume, faces, edges, and vertices.
        """
        data_set = pd.read_csv(cons.PATHS.DATA_MODEL_INPUT / "test.csv")
        match self.model_type:
            case "classifier":
                return data_set[["class_id"]]
            case "regressor":
                return data_set[["volume", "faces", "edges", "vertices"]]
            case _:
                raise ValueError(f"Invalid model type: {self.model_type}")

    def run(self):
        """
        Execute the entire pipeline.
        """
        logger.info("Starting pipeline to output model predictions.")
        # process classifiers
        logger.info("Processing classifier models...")
        classifier_models = self._get_models_by_type("classifier")

        # for each classifier model, load the model, compute predictions and save them to a csv file
        # the csv file has the following columns:
        # - part_id: the id of the part as in the test data set
        # - data_type: the type of data the model was trained on (images, trees or invariants)
        # - pred_class_id: the predicted class id of the part
        predictions = {"path": [], "data_type": [], "pred_class_id": []}

        for model in classifier_models:
            logger.info(f"Processing classifier model: {model}")
            data_type = model.split("-")[0]  # images, trees or invariants
            model = self._load_model(classifier_models[model]["path"])
            test_data_loader = self._get_data_loader("classification", data_type)

            for batch_count, (input_data, target, part_path) in enumerate(
                test_data_loader
            ):
                if batch_count == 2:  # TODO remove early break
                    break
                input_data = input_data.to(device)
                prediction = model(input_data)
                prediction = prediction.cpu().detach()
                prediction = prediction.argmax(dim=1).numpy()
                predictions["path"].extend(part_path)
                predictions["data_type"].extend([data_type] * len(prediction))
                predictions["pred_class_id"].extend(prediction.tolist())
        predictions = pd.DataFrame(predictions)
        predictions.to_csv(
            cons.PATHS.DATA_MODEL_OUTPUT / "classifiers_output.csv", index=False
        )

        # process regressors
        logger.info("Processing regressor models...")
        regressor_models = self._get_models_by_type("regressor")

        predictions = {
            "path": [],
            "data_type": [],
            "pred_volume": [],
            "pred_faces": [],
            "pred_edges": [],
            "pred_vertices": [],
        }
        for model in regressor_models:
            logger.info(f"Processing regressor model: {model}")
            data_type = model.split("-")[0]
            model = self._load_model(regressor_models[model]["path"])
            scaler = self._get_scaler(cons.PATHS.DATA_MODEL_INPUT / "robust_scaler.pkl")
            test_data_loader = self._get_data_loader(
                "regression", data_type, scaler=scaler
            )

            for batch_count, (input_data, target, part_path) in enumerate(
                test_data_loader
            ):
                if batch_count == 2:  # TODO remove early break
                    break
                input_data = input_data.to(device)
                prediction = model(input_data)
                prediction = prediction.cpu().detach().numpy()
                prediction = scaler.inverse_transform(prediction)

                predictions["path"].extend(part_path)
                predictions["data_type"].extend([data_type] * len(prediction))
                predictions["pred_volume"].extend(prediction[:, 0].tolist())
                predictions["pred_faces"].extend(prediction[:, 1].tolist())
                predictions["pred_edges"].extend(prediction[:, 2].tolist())
                predictions["pred_vertices"].extend(prediction[:, 3].tolist())
        predictions = pd.DataFrame(predictions)
        predictions.to_csv(
            cons.PATHS.DATA_MODEL_OUTPUT / "regressors_output.csv", index=False
        )

        logger.info("Pipeline finished.")


if __name__ == "__main__":
    pipeline = ModelsModelOutputPipeline()
    pipeline.run()
