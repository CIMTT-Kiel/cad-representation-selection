"""
Pipeline to generated balanced data splits. (Feature -> Model Input)
"""

# standard libary
import logging
import pickle

# third party packages
import pandas as pd
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

# custom packages
import clearshape.constants as cons
from clearshape.scaler.custom_scalers import LogScaler

# set up logger
logging_level = logging.DEBUG
logger = logging.getLogger(__name__)
logger.setLevel(logging_level)
formatter = logging.Formatter("%(asctime)s %(levelname)8s - %(message)s")
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging_level)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


class FeatureModelInputPipeline:
    """
    Pipeline builds balanced training, validation and test splits based on the data availabel in `4_feature`.

    The pipeline first splits the data set using stratification based on the data points class label, and afterwards balances the training split by oversampling small classes. Oversampling is applied to the training split only, so that no part appears in more than one split.
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

    Notes
    -----
    The assumes the following folder structure:
    data
    |- 4_feature
        |- images
        |   |- fabwave
        |       |- ...
        |- invariants
        |   |- fabwave
        |       |-...
        |- trees
            |- fabwave
                |- ...
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

    # load targets directly from feature csv
    def _get_master_table(self) -> None:
        """
        Initialises the `self._master_table` attribute with all data points available in `fabwave_targets.csv`.
        """
        target_file = cons.PATHS.DATA_FEATURE / "fabwave_targets.csv"
        self._master_table = pd.read_csv(target_file)
        

    def _oversample(self, df: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
        """
        Oversample parts for classes of which there are less than
        `self._class_size_min_required` parts.

        Returns a new DataFrame containing the original rows plus the
        resampled ones. Must only be applied to the training split.

        Parameters
        ----------
        df : pd.DataFrame
            Split to be balanced.
        classes : list[str]
            List of class names that are underrepresented in the split.
        """
        logger.debug("Oversampling small classes.")
        class_sizes = df.value_counts("class_name")
        resampled_parts = []
        for class_name in classes:
            logger.debug(f"Resampling {class_name}")
            original_parts = df.query(f"`class_name` == '{class_name}'")
            n_missing = self._class_size_min_required - class_sizes[class_name]
            if n_missing > 0:
                resampled_parts.append(
                    original_parts.sample(n=n_missing, replace=True, random_state=42)
                )
        return pd.concat([df] + resampled_parts, ignore_index=True)

    def _get_small_classes(self, df: pd.DataFrame) -> list[str]:
        """
        Return list of class names that are underrepresented in the given split.
        """
        class_sizes = df.value_counts("class_name")
        small_classes = class_sizes.index[class_sizes < self._class_size_min_required]
        return small_classes.to_list()

    def _calc_min_required_class_size(self, df: pd.DataFrame) -> int:
        """
        Returns the minimum required class size for the given split.

        The required class size is defined as `0.5 * median_class_size`.
        """
        class_sizes = df.value_counts("class_id")
        return int(class_sizes.median() * 0.5)

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
            stratify=self._master_table["class_id"],
            random_state=42,
        )

        val, test = train_test_split(
            val_and_test,
            test_size=self._conf.test_size
            / (self._conf.test_size + self._conf.val_size),
            stratify=val_and_test["class_id"],
            random_state=42,
        )
        return train, val, test

    def _verify_split_is_balanced(self, df: pd.DataFrame) -> bool:
        """
        Checks whether every class in the given split meets the minimum size.
        """
        class_sizes = df.value_counts("class_id")
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

        # drop classes with too few parts for a stratified split
        class_sizes = self._master_table.value_counts("class_name")
        too_small = class_sizes.index[class_sizes < 10].to_list()
        if too_small:
            logger.info(f"Dropping classes with fewer than 10 parts: {too_small}")
            self._master_table = self._master_table[
                ~self._master_table.class_name.isin(too_small)
            ]

        # generate data splits on unique parts
        logger.info("Generating stratified data splits.")
        train, val, test = self._get_data_splits()

        # verify splits are disjoint
        assert not (set(train.path) & set(test.path)), "overlap"
        assert not (set(train.path) & set(val.path)), "overlap"
        assert not (set(val.path) & set(test.path)), "overla"

        # oversample underrepresented classes in the training split only
        logger.info("Oversampling small classes in training split")
        self._class_size_min_required = self._calc_min_required_class_size(train)
        small_classes = self._get_small_classes(train)
        train = self._oversample(train, small_classes)

        logger.info("Verifying training split is balanced.")
        self._verify_split_is_balanced(train)

        # save log scaler
        logger.info("Fitting and saving log scaler.")
        log_scaler = LogScaler()
        log_scaler.fit(train[["volume", "faces", "edges", "vertices"]]) # just as placeholder method to be consistent with sklearn api
        log_scaler_path = cons.PATHS.DATA_MODEL_INPUT / "log_scaler.pkl"

        # ensure the dir exists
        log_scaler_path.parent.mkdir(exist_ok=True)

        with open(log_scaler_path, "wb") as f:
            pickle.dump(log_scaler, f)

        # save data splits
        logger.info("Saving data splits.")
        train.to_csv(cons.PATHS.DATA_MODEL_INPUT / "train.csv")
        val.to_csv(cons.PATHS.DATA_MODEL_INPUT / "validation.csv")
        test.to_csv(cons.PATHS.DATA_MODEL_INPUT / "test.csv")


if __name__ == "__main__":
    FeatureModelInputPipeline().run()
