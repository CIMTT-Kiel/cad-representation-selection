"""
Pipeline to generated balanced data splits. (Feature -> Model Input)
"""

# standard libary
import logging
from pathlib import Path
from time import sleep

# third party packages
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split

# custom packages
import clearshape.constants as cons

# set up logger
logging_level = logging.INFO
logger = logging.getLogger(__name__)
logger.setLevel(logging_level)
formatter = logging.Formatter("%(levelname)s %(asctime)s - %(message)s")
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging_level)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


class FeatureModelInputPipeline:
    """
    Pipeline builds balanced training, validation and test splits based on the data availabel in `4_feature`.

    The pipeline first balances the data set by oversampling small class, before spliting the data set using stratification based on the data points class label.

    The data splits are saved in `data/5_model_input` as CSV files. Each has the following columns:

    `class`: Name of the class the corresponding part is from.
    `path`: Location path to find data representing the part. The path is relative to any `fabwave` folder containing all the class folders. Also the suffix is striped from each path!

    Expample for data split CSV:
    | class     | path                                          |
    | Bearings  | Bearings/00ed2536-3d80-4f07-8851-4f49f1606498 |

    Methods
    -------
    run()
        Execute the entire pipeline.
    """

    _instance = None

    def __new__(cls):
        """
        Method ensures that only one instance of the pipeline class can exist.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self._conf = OmegaConf.load(
            cons.PATHS.CONFIG / "feature_to_model_input_pipeline.yaml"
        )

    def _get_master_table(self) -> None:
        """
        Initialises the `self._master_table` attribute.
        """
        data = []
        for path in iter((cons.PATHS.DATA_FEATURE / "fabwave").rglob("*.stp")):
            class_name = path.relative_to(
                cons.PATHS.DATA_FEATURE / "fabwave"
            ).parent.as_posix()
            relative_part_path = path.relative_to(
                cons.PATHS.DATA_FEATURE / "fabwave"
            ).as_posix()
            data.append((class_name, relative_part_path))

        self._master_table = pd.DataFrame(data, columns=["class", "path"])

    def _oversample(self, classes: list[str]) -> None:
        """
        Oversample parts for classes of which there are less than
        `self._class_size_min` parts.

        Updates `self._master_table` by randomly drawing parts from the class
        and adding them to the dataset.

        Parameters
        ----------
        classes : list[str]
            List of class name that are underrepresented in the data set.
        """
        class_sizes = self._master_table.value_counts("class")
        for class_name in classes:
            original_parts = self._master_table.query(f"`class` == '{class_name}'")

            # resample class randomly until rquired class size is achived
            logger.debug(f"Resample {class_name}.")
            resampled_parts = []
            for _ in range(self._class_size_min_required - class_sizes[class_name]):
                resampled_part = original_parts.sample(n=1)
                resampled_parts.append(
                    {
                        "class": resampled_part["class"].iloc[0],
                        "path": resampled_part["path"],
                    }
                )

            # add parts of resampled class to master table
            logger.debug("Update master table with resampled parts.")
            self._master_table = pd.concat(
                [self._master_table, pd.DataFrame(resampled_parts)], ignore_index=True
            )

    def _get_small_classes(self) -> list[str]:
        """
        Return list of classes that are underrepresented in the data set.
        """
        class_sizes = self._master_table.value_counts("class")
        small_classes = class_sizes.index[class_sizes < self._class_size_min_required]
        return small_classes.to_list()

    def _calc_min_required_class_size(self) -> int:
        """
        Returns the minimum required class size for each class in the data set.

        The required class size is defined as `0.5 * median_class_size`.
        """
        class_sizes = self._master_table.value_counts("class")
        class_size_median = class_sizes.median()
        class_size_min_required = int(class_size_median * 0.5)
        return class_size_min_required

    def _get_data_splits(self) -> tuple[pd.DataFrame]:
        """
        Returns training, validation and test split.
        """
        # assert split ratios are valid
        assert (
            self._conf.train_size + self._conf.val_size + self._conf.test_size
        ) == 1.0, "Split sizes in config file must sum up to 1."

        train, val_and_test = train_test_split(
            self._master_table,
            train_size=self._conf.train_size,
            stratify=self._master_table["class"],
            random_state=42,
        )

        val, test = train_test_split(
            val_and_test,
            test_size=self._conf.test_size
            / (self._conf.test_size + self._conf.val_size),
            stratify=val_and_test["class"],
            random_state=42,
        )
        return train, val, test

    def _verify_master_table_is_balanced(self):
        """ """
        class_sizes = self._master_table.value_counts("class")
        if class_sizes.min() < self._class_size_min_required:
            logger.warning("The data set has not been balanced correctly!")
            return False
        return True

    def run(self):
        """
        Executes entire pipeline.
        """
        # update pipeline configurations
        logger.info("Updating pipeline configurations.")
        self._conf = OmegaConf.load(
            cons.PATHS.CONFIG / "feature_to_model_input_pipeline.yaml"
        )

        # get identifying paths of all parts along with their class as a dataframe
        logger.info("Initializing master table")
        self._get_master_table()

        # determin classes to oversample
        logger.info("Determining small classes")
        self._class_size_min_required = self._calc_min_required_class_size()
        small_classes = self._get_small_classes()

        # oversample underrepresented classes
        logger.info("Oversampling small classes")
        self._oversample(small_classes)

        # verify master table is balanced now
        logger.info("Verifying data set is balanced.")
        self._verify_master_table_is_balanced()

        # generate data splits
        logger.info("Generating stratified data splits.")
        train, val, test = self._get_data_splits()

        # save data splits
        logger.info("Saving data splits.")
        train.to_csv(cons.PATHS.DATA_MODEL_INPUT / "train.csv")
        val.to_csv(cons.PATHS.DATA_MODEL_INPUT / "validation.csv")
        test.to_csv(cons.PATHS.DATA_MODEL_INPUT / "test.csv")


if __name__ == "__main__":
    FeatureModelInputPipeline().run()
