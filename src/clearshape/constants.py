"""
This module defines several constants used throughout the project.

The module provides access to NamedTupel objects which group all constants and make them available via attribute access.

Examples
--------
>>> from project import constants
>>> # get Path object for the project root directory
>>> constants.PATHS.ROOT
"""
from pathlib import Path

from collections import namedtuple


# Paths
_ROOT = Path(__file__).parents[2]
_path_dict = {
    "ROOT":                 _ROOT,
    "REPORT_FIGURES":       _ROOT / "reports/figures",
    "CONFIG":               _ROOT / "config",

    "DATA_RAW":             _ROOT / "data/1_raw",
    "DATA_INTERMEDIATE":    _ROOT / "data/2_intermediate",
    "DATA_PRIMARY":         _ROOT / "data/3_primary",
    "DATA_FEATURE":         _ROOT / "data/4_feature",
    "DATA_MODEL_INPUT":     _ROOT / "data/5_model_input",
    "DATA_MODELS":          _ROOT / "data/6_models",
    "DATA_MODEL_OUTPUT":    _ROOT / "data/7_model_output",
    "DATA_REPORTING":       _ROOT / "data/8_reporting",

    "MLFLOW_TRACKING_URI": _ROOT / "mlruns",

}



Paths = namedtuple("Paths", list(_path_dict.keys()))
PATHS = Paths(**_path_dict)


# RotationNet costants

_rotnet_dict = {
    "world_size":           1,
    "dist_backend":         'gloo',
    "dist_url":             'tcp://224.66.41.62:23456',
    "distributed":          False,

    "data":                 _ROOT / "Source/clear-shape/data",
    "pretrained":           False, 
    "evaluate":             False, 
    "resume":               '',

    "arch":                 'alexnet',
    "lr":                   0.01,
    "momentum":             0.9,
    "weight_decay":         1e-4,
    "start_epoch":          0,
    "batch_size":           20,
    "workers":              4,
    "epochs":               90,
    
    "print_freq":           10,

}

RotNet = namedtuple("RotNet", list(_rotnet_dict.keys()))
ROTNET = RotNet(**_rotnet_dict)

# API URL
API_URL = "http://step_api:8000/"


# clean up for paths constants
del _path_dict
del Paths
del _ROOT


# general clean up
del namedtuple
del Path