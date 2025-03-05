"""
Pipeline to generate reporting figures for the model input data.

This module provides a pipeline to generate reporting figures for the model
input data. The pipeline loads the model input data, saves descriptive
statistics, and plots class distributions and regression target distributions,
both for the scaled and unscaled data.
"""
# standard libary
import logging
import pickle

# third party packages
import pandas as pd
import seaborn as sns
import seaborn.objects as so
import matplotlib.pyplot as plt

# custom packages
import clearshape.constants as cons

# set up logger
logging_level = logging.DEBUG
logger = logging.getLogger(__name__)
logger.setLevel(logging_level)
formatter = logging.Formatter("%(asctime)s %(levelname)8s - %(message)s")
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging_level)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

sns.set_theme(style="darkgrid", palette="colorblind")

class ModelInputToReportingPipeline:
    """
    Pipeline to generate reporting figures for the model input data.

    Parameters
    ----------
    None

    Attributes
    ----------
    train : pd.DataFrame
        Training data set.
    validation : pd.DataFrame
        Validation data set.
    test : pd.DataFrame
        Test data set.
    scaler : sklearn.preprocessing.RobustScaler
        Scaler used to scale the regression targets.

    Examples
    --------
    >>> pipeline = ModelInputToReportingPipeline()
    >>> pipeline.run()
    """

    def __init__(self) -> None:
        pass

    def _load_data(self) -> None:
        """
        Set `train`, `validation`, and `test` attributes to the data sets loaded from the CSV files in `5_model_input`.

        Returns
        -------
        None
        """
        self.train = pd.read_csv(cons.PATHS.DATA_MODEL_INPUT / "train.csv")
        self.validation = pd.read_csv(cons.PATHS.DATA_MODEL_INPUT / "validation.csv")
        self.test = pd.read_csv(cons.PATHS.DATA_MODEL_INPUT / "test.csv")

    def _save_descriptive_statistics(self) -> None:
        """
        Save descriptive statistics for the `train`, `validation`, and `test`
        data sets as CSV files in `8_reporting/5_model_input`.

        Returns
        -------
        None
        """
        description = self.train.describe(include="all")
        description.to_csv(cons.PATHS.DATA_REPORTING / "5_model_input/train_data_statistics.csv")

        description = self.validation.describe(include="all")
        description.to_csv(cons.PATHS.DATA_REPORTING / "5_model_input/validation_data_statistics.csv")

        description = self.test.describe(include="all")
        description.to_csv(cons.PATHS.DATA_REPORTING / "5_model_input/test_data_statistics.csv")

    def _plot_class_distribution(self) -> None:
        """
        Plot the class distribution for the `train`, `validation`, and `test`
        data sets and save the figures in `8_reporting/5_model_input`.

        The class distribution is visualized as a bar plot.

        Returns
        -------
        None
        """
        for data, name in zip([self.train, self.validation, self.test], ["train", "validation", "test"]):
            figure, axis = plt.subplots(1, 1, figsize=(10, 6))
            plot = (
                so.Plot(data, x="class_name")
                .add(so.Bars(), so.Hist())
                .on(axis)
                )
            axis.tick_params(axis="x", rotation=90)
            figure.tight_layout()
            plot.save(cons.PATHS.DATA_REPORTING / f"5_model_input/{name}_class_distribution.png")

    
    def _plot_regression_target_distributions(self, scale:bool=False) -> None:
        """
        Plot the regression target distributions for the `train`, `validation`, and `test`
        data sets and save the figures in `8_reporting/5_model_input`.
        
        Returns
        -------
        None
        """
        if scale: self._load_scaler()

        for data_name in ["train", "validation", "test"]:
            data = getattr(self, data_name)
            if scale:
                data = self.scaler.transform(data[["volume","faces", "edges", "vertices"]])
                data = pd.DataFrame(data, columns=["volume","faces", "edges", "vertices"])
            figure, axis = plt.subplots(4, 1, figsize=(10, 20))
            for i, target in enumerate(["volume","faces", "edges", "vertices"]):
                plot = (
                    so.Plot(data, x=target)
                    .add(so.Bars(), so.Hist(bins=50))
                    .on(axis[i])
                    )
                
                figure.tight_layout()
                file_name = f"{data_name}_target_distribution_scaled.png" if scale else f"{data_name}_target_distribution.png"
                plot.save(cons.PATHS.DATA_REPORTING / "5_model_input" / file_name)
            
    def _load_scaler(self) -> None:
        """
        Load the scaler from `5_model_input/robust_scaler.pkl` and set it as the `scaler` attribute.
        
        Returns
        -------
        None
        """
        with open(cons.PATHS.DATA_MODEL_INPUT / "robust_scaler.pkl", "rb") as file:
            self.scaler = pickle.load(file)

    def run(self) -> None:
        """
        Executes entire pipeline.

        Returns
        -------
        None
        """
        self._load_data()
        self._save_descriptive_statistics()
        self._plot_class_distribution()
        self._plot_regression_target_distributions()
        self._plot_regression_target_distributions(scale=True) 


if __name__ == "__main__":
    pipeline = ModelInputToReportingPipeline()
    pipeline.run()
        