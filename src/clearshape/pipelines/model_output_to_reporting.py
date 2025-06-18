# TODO: add all missing docstring and update

#%%
# This pipeline is supposed to take the results from the prediction computation and compute the reporting metrics.
# For the classifier models the metrics are accuracy, precision, recall, and F1 score. As well as the confusion matricies.

# The gaol is to save
"""

"""


# standard library
import logging

# third-party
import pandas as pd
import sklearn as skl
from sklearn.utils import multiclass
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so

# custom packages
import clearshape.constants as cons

# set up logger
logging_level = logging.DEBUG
logger = logging.getLogger(__name__)
logger.setLevel(logging_level)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)8s - %(message)s")
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging_level)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

class ModelOutputToReportingPipeline:
    """

    """

    def _filter_for_data_type(self, classifier_output: pd.DataFrame, test_data: pd.DataFrame, data_type: str):
        """
        Filters the classifier output and test data for a specific data type and returns the true class IDs, predicted class IDs, and class names.

        Parameters
        ----------
        classifier_output : pd.DataFrame
            DataFrame containing the classifier output with columns 'data_type', 'path', and 'pred_class_id'.
        test_data : pd.DataFrame
            DataFrame containing the test data with columns 'path', 'class_id', and 'class_name'.
        data_type : str
            The specific data type to filter and process.

        Returns
        -------
        tuple
            A tuple containing the true class IDs, predicted class IDs, and class names.
        """
        data_subset = classifier_output.query("data_type == @data_type")
        # assert values are in correct order
        assert all(data_subset["path"] == test_data["path"])
        class_ids_true = test_data["class_id"]
        class_ids_predicted = data_subset["pred_class_id"]
        return class_ids_true, class_ids_predicted
    
    def _save_confusion_matrix(self, class_ids_true, class_ids_predicted, data_type, class_id_name_map:dict) -> None:
        """
        Calculates and saves confusion matrices for each data type approch.

        The column header of the confusion matricies are the predicted class ids, and the index are the true class ids.

        Parameters
        ----------
        classifier_output : pd.DataFrame
            DataFrame containing the classifier output with columns 'data_type', 'path', and 'pred_class_id'.
        test_data : pd.DataFrame
            DataFrame containing the test data with columns 'path', 'class_id', and 'class_name'.
        
        Returns
        -------
        None
        """
        logger.info(f"Saving confusion matrix for data type: {data_type}")
        confusion_matrix = metrics.confusion_matrix(class_ids_true, class_ids_predicted, normalize='true')

        confusion_matrix = pd.DataFrame(confusion_matrix, index=class_id_name_map.values(), columns=class_id_name_map.values())
        confusion_matrix.to_csv(cons.PATHS.DATA_REPORTING / f"confusion_matrix_{data_type}.csv", index=True)
    
    def _save_classification_report(self, class_ids_true, class_ids_predicted, data_type, class_id_name_map) -> None:
        """
        Calculates and saves the classification report for each data type approach.

        The classification report includes precision, recall, F1-score, and support for each class.

        Parameters
        ----------
        classifier_output : pd.DataFrame
            DataFrame containing the classifier output with columns 'data_type', 'path', and 'pred_class_id'.
        test_data : pd.DataFrame
            DataFrame containing the test data with columns 'path', 'class_id', and 'class_name'.
        
        Returns
        -------
        None
        """
        target_names = list(class_id_name_map.values())
        report = metrics.classification_report(class_ids_true, class_ids_predicted, output_dict=True, target_names=target_names)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(cons.PATHS.DATA_REPORTING / f"classification_report_{data_type}.csv", index=True)

            
    def run(self):
        """
        Execute the pipeline to compute reporting metrics and save confusion matrices.

        Returns
        -------
        None
        """
        logger.info("Starting Model Output to Reporting Pipeline...")
        logger.info("Compute classifier metrics.")
        classifier_output = pd.read_csv(cons.PATHS.DATA_MODEL_OUTPUT / "classifiers_output.csv")
        test_data = pd.read_csv(cons.PATHS.DATA_MODEL_INPUT / "test.csv")

        for data_type in classifier_output["data_type"].unique():
            logger.info(f"Processing data type: {data_type}")
            class_ids_true, class_ids_predicted = self._filter_for_data_type(classifier_output, test_data, data_type)
            class_id_to_class_name = dict(zip(test_data["class_id"].unique(), test_data["class_name"].unique()))
            self._save_confusion_matrix(class_ids_true, class_ids_predicted, data_type, class_id_to_class_name)
            self._save_classification_report(class_ids_true, class_ids_predicted, data_type, class_id_to_class_name)

        logger.info("Compute regression metrics.")
        logger.info("Not implemented yet")
        
        logger.info("Pipeline completed successfully.")

if __name__ == "__main__":
    pipeline = ModelOutputToReportingPipeline()
    pipeline.run()
# %%
