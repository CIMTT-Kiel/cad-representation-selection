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
formatter = logging.Formatter("%(asctime)s %(levelname)8s - %(message)s")
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging_level)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModelsModelOutputPipeline():
    """
    Pipeline class to load models from the `data/6_models` directory and output
    their predictions on the test data set to the `data/model_output` directory.

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

    def _set_models(self):
        """ 
        Sets the models attribute to a dictionary containing the paths to the models.
        """
        self.models = {path.stem: {"path":path} for path in cons.PATHS.DATA_MODELS.iterdir()}

    def _initialize_tree_lstm(self) -> torch.nn.Module:
        """
        Initialize the TreeLSTM model using the best parameters found during the hyperparameter search.

        Returns
        -------
        model : torch.nn.Module
            The TreeLSTM model with randomly initialized weights.
        """
        logger.info(f"Initializing TreeLSTM model")
        
        model_parameter = OmegaConf.load(cons.PATHS.DATA_REPORTING/ f"tree-lstm-{self.model_type}-best-parameter.yaml")
        tuning_pipeline_config = OmegaConf.load(cons.PATHS.CONFIG / f"treelstm_{self.model_type}_tuning_pipeline.yaml")
           
        encoder = RootedInTreeEncoder(
            input_size=tuning_pipeline_config.input_size,   
            encoding_size=model_parameter.encoding_size,                                
            child_sum=True,
        )
        predictor = FeedforwardMLP(
            input_shape=model_parameter.encoding_size,
            hidden_layers=model_parameter.hidden_layers,
            output_shape=tuning_pipeline_config.output_shape,
            task_type=self.model_type,
        )
        model = ModelStack([encoder, predictor])
        logger.debug(self.model_path)
        model.load_state_dict(torch.load(self.model_path))
        model.eval()
        return model    
    
    def _load_model(self) -> None:
        """
        Load the models from the `data/models` directory.

        Returns
        -------
        None
        """
        logger.info(f"Loading model from {self.model_path}")
        # TODO the nacked model has to be generalized (for TreeLSTM, invariant model, and image model)
        # i
        model = None
        match self.data_type:
            case "images":
                pass
            case "invariants":
                pass
            case "trees":
                logger.debug("here")
                model = self._initialize_tree_lstm()
        model.load_state_dict(torch.load(self.model_path))
        model.to(device)
        model.eval()
        self.model = model

    def _set_data_loader(self) -> None:
        """
        Set the DataLoader object for the test data set.

        Returns
        -------
        None
        """
        data_set = FabwaveDataset(cons.PATHS.DATA_MODEL_INPUT / "test.csv", data_type=self.data_type, classification=True, scaler=None)
        match self.data_type:
            case "images":
                data_loader = DataLoader(data_set, batch_size=32, shuffle=False)
            case "invariants":
                data_loader = DataLoader(data_set, batch_size=32, shuffle=False)
            case "trees":
                data_loader = dgl.dataloading.GraphDataLoader(data_set, batch_size=32, shuffle=False)        
        self.data_loader = data_loader
    
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
        predictions = []
        for batch, (input_data, target) in enumerate(self.data_loader):
            input_data = input_data.to(device)
            prediction = self.model(input_data)
            predictions.append(prediction)
        match self.model_type:
            case "classifier":
                predictions = torch.cat(predictions, dim=0).argmax(dim=1).cpu().detach.numpy()
                predictions = pd.DataFrame(predictions, columns=["pred_class_id"])
                assert predictions.shape[1] == 1
            case "regressor":
                columns = self.data_loader.dataset.data.columns[-4:]
                predictions = torch.cat(predictions, dim=0).cpu().detach().numpy()
                # scale predictions
                self._load_scaler()
                predictions = self.scaler.inverse_transform(predictions)
                predictions = pd.DataFrame(predictions, columns=[f"pred_{column}" for column in columns])
                assert predictions.shape[1] == 4

        return predictions

    def _load_scaler(self):
        scaler_path = cons.PATHS.DATA_MODEL_INPUT / "min_max_scaler.pkl"
        with open(scaler_path, "rb") as scaler_file:
            scaler = pickle.load(scaler_file)
        self.scaler = scaler 
        
    def _update_for_next_model(self):
        """
        Set up the pipeline for the next model in the `data/models` directory.

        Returns
        -------
        None
        """
        logger.info("Updating pipeline for next model")
        self.model_path = next(self.model_path_generator)
        self.data_type = self.model_path.stem.split("-")[0]
        self.model_type = self.model_path.stem.split("-")[-1]
        self._load_model()
        self._set_data_loader()

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
        while True:
            try:
                self._update_for_next_model()
            except StopIteration:
                break
                
            predictions = self._compute_predictions()
            # merge predictions with the test data targets
            targets_and_predictions = pd.concat([self._get_targets(), predictions], axis=1)
            targets_and_predictions.to_csv(cons.PATHS.DATA_MODEL_OUTPUT / f"{self.data_type}-{self.model_type}.csv", )

        logger.info("Pipeline completed.")

if __name__ == "__main__":
    pipeline = ModelsModelOutputPipeline()
    pipeline.run()