# %%
"""
This module implements a pipeline to compute the predictions for the data from the test data set
using the models stored in the `data/models` directory.

For more information see the documentation of the `ModelsModelOutputPipeline` class.
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
import torch.nn as nn
from omegaconf import OmegaConf
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

# custom packages
import clearshape.constants as cons
from clearshape.dataset import FabwaveDataset
from clearshape.models.feedforward_mlp import FeedforwardMLP
from clearshape.models.modelstack import ModelStack
from clearshape.models.treelstm import RootedInTreeEncoder
from clearshape.constants import PATHS
from clearshape.models.invariant_mlp import InvariantMLP 
from clearshape.models.trnsfm_encoder import VecsetClassifier, TransformerRegressor
from clearshape.rotationnet.rotnet_classifier import RotationNetModel

# set up logger
logging_level = logging.INFO
logger = logging.getLogger(__name__)
logger.setLevel(logging_level)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)8s - %(message)s")
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging_level)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

torch.manual_seed(42)

class ModelsModelOutputPipeline:
    """
    Pipeline to load trained models from the `data/models` directory and output
    their predictions on the test dataset to the `data/7_model_output` directory.

    This pipeline handles both classifiers and regressors separately. Predictions
    from all classifiers are saved in a single output file, and predictions from
    all regressors are saved in another output file.

    - `data/7_model_output/classifiers_output.csv`
    - `data/7_model_output/regressors_output.csv`

    Parameters
    ----------
    None

    Attributes
    ----------
    model_path_generator : generator
        Generator object to iterate over the files in the `data/models` directory.

    models : dict
        Dictionary containing the paths to the models. The keys are the model file stems
        and the values are dictionaries with the model path.

    Methods
    -------
    run():
        Executes the pipeline to process all models, compute predictions, and save them to output files.

    Notes
    -----
    The pipeline assumes that files containing the models follow this naming
    convention: `{images, trees, or invariants}-{regressor or classifier}.pth`.
    """

    def __init__(self):
        self.model_path_generator = cons.PATHS.DATA_MODELS.iterdir()

    def _get_models_by_type(self, task_type: str):
        """
        Retrieve all models of a specified type (classifier or regressor) from the `data/6_models` directory.

        Parameters
        ----------
        model_type : str
            The type of models to retrieve. Must be either "classifier" or "regressor".

        Returns
        -------
        models : dict
            A dictionary where the keys are the file stems of the models and the values are dictionaries
            containing the paths to the models.

        Raises
        ------
        ValueError
            If the provided `task_type` is not "classifier" or "regressor".

        Notes
        -----
        The naming convention for model files is `{images, trees, or invariants}-{regressor or classifier}.pth`.

        This method filters the models based on their file stem, which is expected to end with either
        "classifier" or "regressor". It returns a dictionary containing the paths to the models of the
        specified type.
        """
        if task_type not in {"classifier", "regressor"}:
            raise ValueError("model_type must be either 'classifier' or 'regressor'")

        models = {}
        for path in cons.PATHS.DATA_MODELS.iterdir():
            if path.stem.endswith(task_type):
                models[path.stem] = {"path": path}

        return models

    def _initialize_tree_lstm(self, task_type) -> torch.nn.Module:
        """
        Initialize a TreeLSTM model for either classification or regression tasks.

        This method loads the best hyperparameters for the TreeLSTM model from a YAML file
        and constructs the model using a RootedInTreeEncoder for encoding tree-structured data
        and a FeedforwardMLP for prediction. The model is designed to handle either classification
        or regression tasks based on the provided `task_type`.

        Parameters
        ----------
        task_type : str
            The type of task for which the TreeLSTM model is being initialized.
            Must be either "classifier" or "regressor".

        Returns
        -------
        torch.nn.Module
            A TreeLSTM model composed of a RootedInTreeEncoder and a FeedforwardMLP
            predictor, with randomly initialized weights.

        Notes
        -----
        - The method assumes that the best hyperparameters for the model are stored in a YAML file
          named `tree-lstm-{task_type}-best-parameter.yaml` in the `data/6_models` directory.
        - The tuning pipeline configuration is loaded from a YAML file named
          `treelstm_{task_type}_tuning_pipeline.yaml` in the `config` directory.
        - The `RootedInTreeEncoder` is used for encoding tree-structured data, and the `FeedforwardMLP`
          is used as the predictor.
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
            output_shape=pd.read_csv(cons.PATHS.DATA_MODEL_INPUT / "train.csv")["class_id"].nunique() if task_type == "classifier" else tuning_pipeline_config.output_shape,
            dropout_rate=model_parameter.dropout_rate,
            activation=nn.ReLU(),
            output_activation=nn.Softmax(dim=1) if task_type == "classifier" else nn.ReLU(),
        )
        model = ModelStack([encoder, predictor])
        return model

    # TODO: Implement the image model initialization
    def _initialize_image_model(self, task_type) -> torch.nn.Module:
        """
        Initialize the RotationNetModel model for classification tasks.

        This method loads the best trained checkpoint for the image model from a saved
        `.ckpt` file, extracts the hyperparameters and state dictionary, adapts the state
        dictionary keys if necessary, and instantiates the model with the loaded parameters.

        Returns
        -------
        torch.nn.Module
            An InvariantMLP model initialized with the best saved weights and hyperparameters.

        Notes
        -----
        - The checkpoint is assumed to be stored in the `data/6_models/invariants_classification.ckpt` file.
        - The hyperparameter dictionary inside the checkpoint contains parameters used to
        instantiate the `InvariantMLP`. The learning rate (`lr`) key is removed before model initialization.
        - The keys in the `state_dict` may have a "model." prefix, which is stripped before loading.
        - This method assumes the model class `InvariantMLP` is already imported and available.
        """
        #TODO implement for regression

        logger.info(f"Initializing Images-model")

        checkpoint_path = PATHS.DATA_MODELS / "images-classifier-v1.ckpt"
        # hyperparams = checkpoint["hyper_parameters"]
        # del hyperparams["lr"]

        model = RotationNetModel.load_from_checkpoint(checkpoint_path)#**hyperparams)

        return model

    # TODO: Implement the invariant model initialization
    def _initialize_invariant_model(self, task_type) -> torch.nn.Module:
        """
        Initialize the InvariantMLP model for classification tasks.

        This method loads the best trained checkpoint for the invariant model from a saved
        `.ckpt` file, extracts the hyperparameters and state dictionary, adapts the state
        dictionary keys if necessary, and instantiates the model with the loaded parameters.

        Returns
        -------
        torch.nn.Module
            An InvariantMLP model initialized with the best saved weights and hyperparameters.

        Notes
        -----
        - The checkpoint is assumed to be stored in the `data/6_models/invariants_classification.ckpt` file.
        - The hyperparameter dictionary inside the checkpoint contains parameters used to
        instantiate the `InvariantMLP`. The learning rate (`lr`) key is removed before model initialization.
        - The keys in the `state_dict` may have a "model." prefix, which is stripped before loading.
        - This method assumes the model class `InvariantMLP` is already imported and available.
        """
        #TODO implement for regression

        logger.info(f"Initializing Invariants-model")

        checkpoint = torch.load((PATHS.DATA_MODELS / "invariants-classifier.ckpt").as_posix())
        hyperparams = checkpoint["hyper_parameters"]
        del hyperparams["lr"]

        model = InvariantMLP(**hyperparams)

        return model
    
    def _initialize_vecset_model(self, task_type) -> torch.nn.Module:
        """
        Initialize a VecSet model for either classification or regression tasks.

        This method loads a pre-trained VecSet model from a checkpoint file, reconstructs the model 
        using the saved hyperparameters (excluding the learning rate), und loads the trained weights. 
        The method is used to initialize the VecSet-based model architecture for downstream tasks such 
        as classification or regression.

        Parameters
        ----------
        task_type : str
            The type of task for which the VecSet model is being initialized.
            Typically "classifier" or "regressor". (Note: currently not used to branch logic.)

        Returns
        -------
        torch.nn.Module
            A VecSetClassifier model with architecture and weights restored from the checkpoint.

        Notes
        -----
        - The model checkpoint is expected at `data/6_models/vecsets-classifierckpt.ckpt`.
        - The checkpoint must contain a `"state_dict"` with weight tensors and `"hyper_parameters"` 
        used to define the model architecture.
        - The `"lr"` hyperparameter is excluded during reconstruction, as it is optimizer-specific.
        - The keys in the saved `state_dict` are expected to be prefixed with `"model."`, which will
        be stripped before loading into the instantiated model.
        - The model class `VecsetClassifier` must be importable in the current environment.
        """
        #TODO implement for regression
        logger.info(f"Initializing Vecset-model")

        if task_type == "classifier":

            checkpoint = torch.load((PATHS.DATA_MODELS / "vecsets-classifier.ckpt").as_posix())
            hyperparams = checkpoint["hyper_parameters"]
            del hyperparams["lr"]
            del hyperparams["weight_decay"]

            model = VecsetClassifier(**hyperparams)
        
        elif task_type == "regressor":
            checkpoint = torch.load((PATHS.DATA_MODELS / "vecsets-regressor.ckpt").as_posix())
            hyperparams = checkpoint["hyper_parameters"]
            del hyperparams["lr"]
            del hyperparams["weight_decay"]
            del hyperparams["max_epochs"]
            del hyperparams["warmup_epochs"]
            del hyperparams["target_names"]
            

            model = TransformerRegressor(**hyperparams)

            


        return model

    def _load_model(self, path) -> None:
        """
        Load a model from the specified path and initialize it based on its type.

        This method determines the type of model (classifier or regressor) and the
        data type (images, trees, or invariants) from the filename, initializes the
        appropriate model, and loads its state dictionary.

        Parameters
        ----------
        path : pathlib.Path
            Path to the model's state dictionary file. The filename should follow
            the format `<data_type>-<model_type>.pt`, where `data_type` can be
            'images', 'trees', or 'invariants', and `model_type` can be 'classifier'
            or 'regressor'.

        model : torch.nn.Module
            The initialized and loaded PyTorch model.

        Raises
        ------
        ValueError
            If the `data_type` or `model_type` extracted from the filename is invalid.
        """
        logger.debug(f"Entering '_load_model' with path: {path}")
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
            case ("vecsets", "classifier"):
                model = self._initialize_vecset_model("classifier")
            case ("vecsets", "regressor"):
                model = self._initialize_vecset_model("regressor")
        if model is None:
            raise ValueError(
                f"Invalid model type: {data_type}-{model_type}. Supported types are: images, trees, invariants for data and regressor, classifier for model type."
            )

        if path.suffix == ".pth":
            state_dict = torch.load(path, weights_only=False, map_location=device)
        elif path.suffix == ".ckpt":
            checkpoint = torch.load(path.as_posix(), map_location=device)
            state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
        else:
            raise NotImplementedError(f"Loading model format {path.suffix} is not implemented")

        model.to(device)
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        return model

    def _get_data_loader(self, task_type, data_type, scaler=None) -> None:
        """
        Creates and returns a DataLoader object for the test dataset based on the specified task type and data type.

        Parameters
        ----------
        task_type : str
            The type of task to perform. Supported values are "regression" and "classification".
        data_type : str
            The type of data to process. Supported values are "images", "invariants", and "trees".
        scaler : object, optional
            A scaler object to normalize the data, if applicable. Default is None.

        Returns
        -------
        DataLoader or GraphDataLoader
            A DataLoader object for "images" and "invariants" data types, or a GraphDataLoader object for "trees" data type.
        """
        logger.debug("Calling'_get_data_loader'.")
        data_set = FabwaveDataset(
            cons.PATHS.DATA_MODEL_INPUT / "test.csv",
            data_type=data_type,
            regression=task_type == "regression",
            classification=task_type == "classification",
            scaler=scaler,
        )
        match data_type:
            case "images":
                data_loader = DataLoader(data_set, batch_size=20, shuffle=False)
            case "invariants":
                data_loader = DataLoader(data_set, batch_size=256, shuffle=False)
            case "vecsets":
                data_loader = DataLoader(data_set, batch_size=256, shuffle=False)
            case "trees":
                data_loader = dgl.dataloading.GraphDataLoader(
                    data_set, batch_size=256, shuffle=False, 
                )
        return data_loader

    def _get_scaler(self, path):
        """
        Load a scaler object from a specified file path.

        This method reads a file containing a serialized scaler object
        and deserializes it using the `pickle` module.

        Parameters
        ----------
        path : str
            The file path to the serialized scaler object.

        Returns
        -------
        object
            The deserialized scaler object.
        """
        logger.debug("Calling '_get_scaler'.")
        with open(path, "rb") as scaler_file:
            return pickle.load(scaler_file)

    def _process_batch_in_chunks(self, model, input_data, chunk_size=32):
        """
        Process a large batch in smaller chunks to avoid GPU memory overflow.

        This method splits a large input batch into smaller sub-batches,
        processes each sub-batch separately, and concatenates the results.

        Parameters
        ----------
        model : torch.nn.Module
            The model to use for prediction.
        input_data : torch.Tensor
            The input data batch to process.
        chunk_size : int, optional
            The size of each sub-batch. Default is 32.

        Returns
        -------
        torch.Tensor
            The concatenated predictions from all sub-batches.
        """
        predictions = []
        num_samples = input_data.shape[0]

        for i in range(0, num_samples, chunk_size):
            end_idx = min(i + chunk_size, num_samples)
            chunk = input_data[i:end_idx]

            with torch.no_grad():
                chunk_pred = model(chunk)
                predictions.append(chunk_pred.cpu())

            # Clear GPU cache after each chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return torch.cat(predictions, dim=0)

    def run(self):
        """
        Execute the pipeline to generate predictions from classifier
        and regressor models, saving the results to CSV files.
        """
        logger.info("Starting pipeline to output model predictions.")

        # process classifiers
        logger.info("=== Processing classifier models ===")
        classifier_models = self._get_models_by_type("classifier")

        # for each classifier model, load the model, compute predictions and save them to a csv file
        # the csv file has the following columns:
        # - part_id: the id of the part as in the test data set
        # - data_type: the type of data the model was trained on (images, trees or invariants)
        # - pred_class_id: the predicted class id of the part
        predictions = {"path": [], "data_type": [], "pred_class_id": []}

        for model in classifier_models:
            logger.info(f"Computing output for {model} model")
            model_name = model
            data_type = model.split("-")[0]  # images, trees or invariants
            model = self._load_model(classifier_models[model]["path"])
            test_data_loader = self._get_data_loader("classification", data_type)

            for batch_count, (input_data, target, part_path) in enumerate(
                test_data_loader
            ):

                input_data = input_data.to(device)
                if model_name == 'images-classifier':
                    prediction = model.predict_step(input_data)
                    prediction = prediction.cpu().detach()
                elif model_name == 'vecsets-classifier':
                    # Use chunking for vecsets-classifier to avoid GPU memory issues
                    prediction = self._process_batch_in_chunks(model, input_data, chunk_size=32)
                    prediction = prediction.detach()
                    prediction = prediction.argmax(dim=1).numpy()
                else:
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
        logger.info("=== Processing regressor models ===")
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
            logger.info(f"Computing output for {model} model")
            data_type = model.split("-")[0]
            model = self._load_model(regressor_models[model]["path"])
            scaler = self._get_scaler(cons.PATHS.DATA_MODEL_INPUT / "log_scaler.pkl")
            test_data_loader = self._get_data_loader(
                "regression", data_type, scaler=scaler
            )

            for batch_count, (input_data, target, part_path) in enumerate(
                test_data_loader
            ):

                input_data = input_data.to(device)

                # Use chunking for vecsets-regressor to avoid GPU memory issues
                if data_type == 'vecsets':
                    prediction = self._process_batch_in_chunks(model, input_data, chunk_size=32)
                    prediction = prediction.detach().numpy()
                else:
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
