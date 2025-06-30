# %%
"""
This module implements a pipeline to process the predictions and the test data to compute reporting metrics.

For details see the documentation of the `ModelOutputToReportingPipeline` class.
"""

# standard library
import logging

# third-party
import pandas as pd
import sklearn.metrics as metrics

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
    Pipeline to process the model output and test data to compute reporting metrics.

    This pipeline computes confusion matrices for classification tasks, classification metrics,
    regression metrics, and an error table for regression tasks. It saves the results to CSV files
    in the reporting directory.

    Parameters
    ----------
    None

    Attributes
    ----------
    None

    Methods
    -------
    run() : None
        Executes the pipeline to compute reporting metrics and save confusion matrices.
    """

    def _get_true_and_prediced_values(
        self,
        output: pd.DataFrame,
        test_data: pd.DataFrame,
        data_type: str,
        is_classifier: bool,
    ):
        """
        Filters the model output for a specific data type and returns the true and predicted values.

        This method ensures that the paths in the model output and test data match for the specified data type.
        For classification tasks, it retrieves the true class IDs and predicted class IDs.
        For regression tasks, it retrieves the true and predicted values for the attributes volume, faces, edges, and vertices.

        Parameters
        ----------
        output : pd.DataFrame
            DataFrame containing the model output with columns 'data_type', 'path', and prediction columns.
        test_data : pd.DataFrame
            DataFrame containing the test data with columns 'path' and true value columns.
        data_type : str
            The specific data type to filter and process. (trees, images, or invariants)
        is_classifier : bool
            Whether the output is from a classifier (True) or regressor (False).

        Returns
        -------
        tuple
            A tuple containing the true values and predicted values as pandas Series or DataFrames.
        """
        logger.info(f"Filtering output for data type: {data_type}")
        data_subset = output.query("data_type == @data_type")
        assert all(data_subset["path"] == test_data["path"])

        if is_classifier:
            true_values = test_data["class_id"]
            predicted_values = data_subset["pred_class_id"]
        else:
            true_values = test_data[["volume", "faces", "edges", "vertices"]]
            predicted_values = data_subset[
                ["pred_volume", "pred_faces", "pred_edges", "pred_vertices"]
            ]

        return true_values, predicted_values

    def _get_confusion_matrix(
        self, class_ids_true, class_ids_predicted, data_type, class_id_name_map: dict
    ) -> None:
        """
        Saves confusion matrices for each data type approach.

        The confusion matrix provides normalized values representing the proportion of true class IDs classified as each predicted class ID.
        The rows correspond to true class IDs, and the columns correspond to predicted class IDs.

        Parameters
        ----------
        class_ids_true : pd.Series
            Series containing the true class IDs.
        class_ids_predicted : pd.Series
            Series containing the predicted class IDs.
        data_type : str
            The specific data type being processed (e.g., trees, images, or invariants).
        class_id_name_map : dict
            Dictionary mapping class IDs to their corresponding class names.

        Returns
        -------
        None
        """
        logger.info(f"Saving confusion matrix for data type: {data_type}")
        confusion_matrix = metrics.confusion_matrix(
            class_ids_true, class_ids_predicted, normalize="true"
        )

        confusion_matrix = pd.DataFrame(
            confusion_matrix,
            index=class_id_name_map.values(),
            columns=class_id_name_map.values(),
        )
        return confusion_matrix

    def _get_classification_metrics(
        self, classifier_output: pd.DataFrame, test_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Computes and saves classification metrics for each data type.

        This method calculates metrics such as accuracy, F1-score, recall, and precision for classification tasks.
        The results are aggregated and saved as a CSV file in the reporting directory.

        Parameters
        ----------
        classifier_output : pd.DataFrame
            DataFrame containing the classifier output with columns 'data_type', 'path', and 'pred_class_id'.
        test_data : pd.DataFrame
            DataFrame containing the test data with columns 'path', 'class_id', and 'class_name'.

        Returns
        -------
        results_df : pd.DataFrame
            DataFrame containing the classification metrics for each data type.
            The DataFrame has columns 'data_type', 'accuracy_micro', 'f1_score_micro',
            'recall_micro', and 'precision_micro'.
        """
        logger.debug("Saving classification metrics")
        results = []
        for data_type in classifier_output["data_type"].unique():
            class_ids_true, class_ids_predicted = self._filter_output_for_data_type(
                classifier_output, test_data, data_type, is_classifier=True
            )
            accuracy = metrics.accuracy_score(class_ids_true, class_ids_predicted)
            f1_score = metrics.f1_score(
                class_ids_true, class_ids_predicted, average="micro"
            )
            recall = metrics.recall_score(
                class_ids_true, class_ids_predicted, average="micro"
            )
            precision = metrics.precision_score(
                class_ids_true, class_ids_predicted, average="micro"
            )
            results.append(
                {
                    "data_type": data_type,
                    "accuracy_micro": accuracy,
                    "f1_score_micro": f1_score,
                    "recall_micro": recall,
                    "precision_micro": precision,
                }
            )

        results_df = pd.DataFrame(results)
        return results_df

    def _get_regression_metrics(
        self, regressor_output: pd.DataFrame, test_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculates and returns the regression metrics for each data type approach.

        The regression metrics include Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared.

        Parameters
        ----------
        regressor_output : pd.DataFrame
            DataFrame containing the regressor output with columns 'data_type', 'path', and predicted values.
        test_data : pd.DataFrame
            DataFrame containing the test data with columns 'path' and true values.

        Returns
        -------
        results_df : pd.DataFrame
            DataFrame containing the regression metrics for each data type and attribute.
            The DataFrame has columns 'data_type', 'attribute', 'mae', 'mse', and 'r2'.
        """
        logger.debug("Getting regression metrics")
        results = []
        for data_type in regressor_output["data_type"].unique():
            for attribute in ["volume", "faces", "edges", "vertices"]:
                true_values, predicted_values = self._filter_output_for_data_type(
                    regressor_output, test_data, data_type, is_classifier=False
                )
                mae = metrics.mean_absolute_error(true_values, predicted_values)
                mse = metrics.mean_squared_error(true_values, predicted_values)
                r2 = metrics.r2_score(true_values, predicted_values)
                results.append(
                    {
                        "data_type": data_type,
                        "attribute": attribute,
                        "mae": mae,
                        "mse": mse,
                        "r2": r2,
                    }
                )

            results_df = pd.DataFrame(results)
        return results_df

    def _get_error_table(
        self, regressor_output: pd.DataFrame, test_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculates and returns the error table for regression metrics.

        The error table includes the absolute error, relative error, and percentage error for each attribute.

        Parameters
        ----------
        regressor_output : pd.DataFrame
            DataFrame containing the regressor output with columns 'data_type', 'path', and predicted values.
        test_data : pd.DataFrame
            DataFrame containing the test data with columns 'path', 'volume', 'faces', 'edges', and 'vertices'.

        Returns
        -------
        errors : pd.DataFrame
            DataFrame containing the error table with columns 'path', 'data_type', 'volume_error
            ', 'faces_error', 'edges_error', 'vertices_error', 'volume_relative_error',
            'faces_relative_error', 'edges_relative_error', and 'vertices_relative_error'.
        """
        logger.info("Calculating error table")
        errors = pd.DataFrame(
            columns=[
                "path",
                "data_type",
                "volume_error",
                "faces_error",
                "edges_error",
                "vertices_error",
            ]
        )
        for data_type in regressor_output["data_type"].unique():
            data_type_output = regressor_output.query("data_type == @data_type")
            errors = pd.concat(
                [errors, data_type_output[["path", "data_type"]]], ignore_index=True
            )
            errors["volume_error"] = abs(data_type_output["pred_volume"] - test_data["volume"])
            errors["faces_error"] = abs(data_type_output["pred_faces"] - test_data["faces"])
            errors["edges_error"] = abs(data_type_output["pred_edges"] - test_data["edges"])
            errors["vertices_error"] = abs(data_type_output["pred_vertices"] - test_data["vertices"])
            errors["volume_relative_error"] = (errors["volume_error"] / test_data["volume"])
            errors["faces_relative_error"] = errors["faces_error"] / test_data["faces"]
            errors["edges_relative_error"] = errors["edges_error"] / test_data["edges"]
            errors["vertices_relative_error"] = (errors["vertices_error"] / test_data["vertices"])
        return errors
        )

    def _save_classification_metrics_plot(self, classification_metrics: pd.DataFrame) -> None:
        """
        Creates and saves a bar plot for the classification metrics.

        This method generates a bar plot showing the accuracy, F1-score, recall, and precision
        for each data type. The plot is saved as a PNG file in the reporting directory.

        Parameters
        ----------
        classification_metrics : pd.DataFrame
            DataFrame containing the classification metrics with columns 'data_type', 'accuracy_micro',
            'f1_score_micro', 'recall_micro', and 'precision_micro'.

        Returns
        -------
        so.Plot
            A seaborn objects Plot instance representing the bar plot.
        """
        plot = (
            so.Plot(classification_metrics, x="data_type")
            .add(so.Bar(), so.Dodge())
            .label(
                title="Classification Metrics by Data Type",
                x="Data Type",
                y="Metric Value",
                color="Metric",
            )
            .scale(y=so.Scale("linear", zero=False))
        )
        plot.save(
            cons.PATHS.DATA_REPORTING / "classification_metrics_plot.png",
            format="png",
            bbox_inches="tight",
        )
    def run(self):
        """
        Execute the pipeline to compute reporting metrics and save confusion matrices.

        Returns
        -------
        None
        """
        logger.info("Starting Model Output to Reporting Pipeline...")

        # === CLASSIFICATION METRICS ===
        logger.info("Compute classifier metrics.")
        # load predictions and test data
        classifier_output = pd.read_csv(
            cons.PATHS.DATA_MODEL_OUTPUT / "classifiers_output.csv"
        )
        test_data = pd.read_csv(cons.PATHS.DATA_MODEL_INPUT / "test.csv")[
            ["path", "class_id", "class_name"]
        ]

        # calculate and save confusion matrices for each data type
        for data_type in classifier_output["data_type"].unique():
            logger.info(f"Processing data type: {data_type}")
            class_ids_true, class_ids_predicted = self._filter_output_for_data_type(
                classifier_output, test_data, data_type, is_classifier=True
            )
            class_id_to_class_name = dict(
                zip(test_data["class_id"].unique(), test_data["class_name"].unique())
            )
            self._save_confusion_matrix(
                class_ids_true, class_ids_predicted, data_type, class_id_to_class_name
            )

        self._save_classification_metrics(classifier_output, test_data)

        # === REGRESSION METRICS ===
        logger.info("Compute regression metrics.")
        # load predictions and test data
        regressor_output = pd.read_csv(
            cons.PATHS.DATA_MODEL_OUTPUT / "regressors_output.csv"
        )
        test_data = pd.read_csv(cons.PATHS.DATA_MODEL_INPUT / "test.csv")[
            ["path", "volume", "faces", "edges", "vertices"]
        ]

        self._save_regression_metrics(regressor_output, test_data)
        self._calc_error_table(regressor_output, test_data)

        logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    pipeline = ModelOutputToReportingPipeline()
    pipeline.run()
# %%
