"""

"""

# standard libary
import typing
from typing import Union
from pathlib import Path
import pickle
import logging

# third party libaries
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import dgl
import json

# custom packages
import clearshape.constants as cons

# set up logger
logging_level = logging.INFO
logger = logging.getLogger(__name__)
logger.setLevel(logging_level)
formatter = logging.Formatter("%(asctime)s %(levelname)8s - %(message)s")
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging_level)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

class FabwaveDataset(Dataset):
    """
    A PyTorch dataset class to load data from a CSV file containing
    classification and regression features.

    Parameters
    ----------
    csv_file : str
        Path to the CSV file containing the entries for the data split to be used.
    data_type : str
        "images", "trees" or "invariants" to specify the data type.
    task_type : str, optional
        The type of task ('classification' or 'regression'). Default is
        'classification'.
    transform : callable, optional
        Optional transform to be applied on a sample.
    """

    def __init__(self, csv_file: Union[str, Path], data_type: str, regression:bool=False, classification:bool=False, scaler=None,):
        assert data_type in ["images", "trees", "invariants", "vecsets"], "Is the data type spelled correctly?"
        assert regression ^ classification, "Please specify the task type: 'classification' or 'regression'."
        self.scaler = scaler  
        self.data = self._get_scaled_data(csv_file) if self.scaler else self._load_data(csv_file) # Load CSV file into a DataFrame
        self.data_type = data_type
        self.regression = regression
        self.classification = classification

        #default for invariants
        self.inv_only = True

    def _load_data(self, csv_file: Union[str, Path]) -> pd.DataFrame:
        return pd.read_csv(csv_file, index_col=0)
    
    def _get_scaled_data(self, csv_file: Union[str, Path]) -> pd.DataFrame:
        data = self._load_data(csv_file)
        data[["volume", "faces", "edges", "vertices"]] = self.scaler.transform(data[["volume", "faces", "edges", "vertices"]])
        return data

    def __len__(self):
        """
        Returns the total number of samples.

        Returns
        -------
        int
            Number of samples in the dataset.
        """
        return len(self.data)  # Return the number of rows in the DataFrame

    def __getitem__(self, idx):
        """
        Retrieves a sample at the given index.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        dict
            A dictionary containing the features and label.
        """
        row = self.data.iloc[idx]

        # Determine the correct folder based on the extracted folder name
        
        file_paths = {
            'images': cons.PATHS.DATA_FEATURE / 'images/fabwave' / f"{row.name}.png",
            'trees': cons.PATHS.DATA_FEATURE  / 'trees/fabwave' / f"{row.name}.bin",
            'invariants': cons.PATHS.DATA_FEATURE / 'invariants/fabwave' / f"{row.name}.json",
            'vecsets': cons.PATHS.DATA_FEATURE / 'vecsets/fabwave' / f"{row.name}.npy"
        }

        if self.data_type not in file_paths:
            raise ValueError(f"Unknown data category in path: {row.name}")

        file_path = file_paths[self.data_type]

        # Load the CAD model representation; images, trees or invariants
        if self.data_type == 'trees':
            data_representation = dgl.load_graphs(file_path.as_posix())[0][0]
        elif self.data_type == 'invariants':
            with open(file_path.as_posix(), 'r') as f:
                invs = json.load(f)

                #delete pi_200, pi_020 and pi_002
                del invs['pi_200']
                del invs['pi_020']
                del invs['pi_002']

                # split into moments and invariants
                mue_values = [] #moments
                pi_values = [] # invariants

                for key, value in invs.items():
                    if key.startswith('mue_'):
                        mue_values.append(value)
                    elif key.startswith('pi_'):
                        pi_values.append(value)

                # TODO: scale the moments if used in combination with invariants

                if self.inv_only:
                    data_representation = torch.tensor(pi_values, dtype=torch.float32)
                else:
                    data_representation = torch.tensor(mue_values + pi_values, dtype=torch.float32)

        elif self.data_type == 'vecsets':
            # Load the vecset from a .npy file
            data_representation = torch.Tensor(np.load(file_path.as_posix(), allow_pickle=True))

        elif self.data_type == 'images':
            raise NotImplementedError("Image data type not implemented yet.")
            # data_representation = 
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")  # Raise error for unsupported formats
        
        # get task-specific targets
        if self.classification:
            target = torch.tensor(row['class_id'])
        elif self.regression:
            target = torch.tensor([row['volume'], row['faces'], row['edges'], row['vertices']], dtype=torch.float32)  # Convert class label to float for regression
        
        path = row.name
        return data_representation, target, path
    


