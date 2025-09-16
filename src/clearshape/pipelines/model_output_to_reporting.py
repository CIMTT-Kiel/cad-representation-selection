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
        #assert all(data_subset["path"] == test_data["path"])

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
        
        class_id_name_map = dict(sorted(class_id_name_map.items()))
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
            The DataFrame has columns 'data_type', 'metric', and 'value'.
        """
        logger.debug("Saving classification metrics")
        results = []
        for data_type in classifier_output["data_type"].unique():
            class_ids_true, class_ids_predicted = self._get_true_and_prediced_values(
                classifier_output, test_data, data_type, is_classifier=True
            )
            logger.debug(class_ids_true)
            logger.debug("")
            logger.debug(class_ids_predicted)
            accuracy = metrics.accuracy_score(class_ids_true, class_ids_predicted)
            f1_score = metrics.f1_score(
                class_ids_true, class_ids_predicted, average="macro"
            )
            recall = metrics.recall_score(
                class_ids_true, class_ids_predicted, average="macro"
            )
            precision = metrics.precision_score(
                class_ids_true, class_ids_predicted, average="macro"
            )
            results.extend(
                [
                    {"data_type": data_type, "metric": "accuracy_macro", "value": accuracy},
                    {"data_type": data_type, "metric": "f1_score_macro", "value": f1_score},
                    {"data_type": data_type, "metric": "recall_macro", "value": recall},
                    {"data_type": data_type, "metric": "precision_macro", "value": precision},
                ]
            )
        results_df = pd.DataFrame(results)
        return results_df

    def _get_regression_metrics(
        self, regressor_output: pd.DataFrame, test_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Computes regression metrics (MAE, MSE, R-squared) for each data type and attribute.

        Parameters
        ----------
        regressor_output : pd.DataFrame
            DataFrame containing the regressor output with columns 'data_type', 'path', and predicted values.
        test_data : pd.DataFrame
            DataFrame containing the test data with columns 'path' and true values.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns 'data_type', 'attribute', 'metric', and 'value',
            summarizing regression metrics for each data type and attribute.
        """
        logger.debug("Calculating regression metrics")
        metrics_list = []

        for data_type in regressor_output["data_type"].unique():
            true_values, predicted_values = self._get_true_and_prediced_values(
                regressor_output, test_data, data_type, is_classifier=False
            )

            for attribute in ["volume", "faces", "edges", "vertices"]:
                attribute_true = true_values[attribute]
                attribute_pred = predicted_values[f"pred_{attribute}"]

                metrics_list.extend(
                    [
                        {"data_type": data_type, "attribute": attribute, "metric": "mae", "value": metrics.mean_absolute_error(attribute_true, attribute_pred)},
                        {"data_type": data_type, "attribute": attribute, "metric": "mse", "value": metrics.mean_squared_error(attribute_true, attribute_pred)},
                        {"data_type": data_type, "attribute": attribute, "metric": "r2", "value": metrics.r2_score(attribute_true, attribute_pred)},
                    ]
                )

        return pd.DataFrame(metrics_list)
    
    def _save_regression_metrics_plot(self, regression_metrics:pd.DataFrame) -> None:
        """
        Builds and saves plot which compares regression metrics accross all approaches and attributes.

        MAE, R2 and MSE for each attribute and values grouped by data type.
        """
        plot = (
            so.Plot(regression_metrics, x="metric", y="value", color="data_type", )
            .facet("attribute")
            .add(so.Bar(), so.Dodge())
            .label(
                x="Metric",
                y="Value",
                color="Data Type",
            )
            .layout(size=(15, 5))
        )
        plot.save(
            cons.PATHS.DATA_REPORTING / "regression_metrics_plot.png",
            format="png",
            bbox_inches="tight",
        )

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
            DataFrame containing the error table with columns 'path', 'data_type', 'error_type', 'value'
        """
        logger.info("Calculating error table")
        # for each data type and each cad model indentified by the path, calculate the relative error for each attribute
        errors = []
        for data_type in regressor_output["data_type"].unique():
            data_type_output = regressor_output.query("data_type == @data_type")
            for _, row in data_type_output.iterrows():
                true_values = test_data.query("path == @row['path']")
                if true_values.empty:
                    continue
                true_values = true_values.iloc[0]
                # calculate absolute errors
                volume_error = abs(row["pred_volume"] - true_values["volume"])
                faces_error = abs(row["pred_faces"] - true_values["faces"])
                edges_error = abs(row["pred_edges"] - true_values["edges"])
                vertices_error = abs(row["pred_vertices"] - true_values["vertices"])
                # calculate relative errors
                volume_relative_error = volume_error / true_values["volume"] if true_values["volume"] != 0 else 0
                faces_relative_error = faces_error / true_values["faces"] if true_values["faces"] != 0 else 0
                edges_relative_error = edges_error / true_values["edges"] if true_values["edges"] != 0 else 0
                vertices_relative_error = vertices_error / true_values["vertices"] if true_values["vertices"] != 0 else 0
                # append errors to the list
                errors.append(
                    {
                        "path": row["path"],
                        "data_type": data_type,
                        "volume_error": volume_error,
                        "faces_error": faces_error,
                        "edges_error": edges_error,
                        "vertices_error": vertices_error,
                        "volume_relative_error": volume_relative_error,
                        "faces_relative_error": faces_relative_error,
                        "edges_relative_error": edges_relative_error,
                        "vertices_relative_error": vertices_relative_error,
                    }
                )
        # create a DataFrame from the errors list
        errors = pd.DataFrame(errors)
        # melt the DataFrame to have a long format with error types
        errors = errors.melt(
            id_vars=["path", "data_type"],
            var_name="error_type",
            value_name="value",
        )
        return errors

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
            so.Plot(classification_metrics, x="metric", y="value", color="data_type")
            .add(so.Bar(), so.Dodge())
            .label(
                title="Classification Metrics by Data Type",
                x="Metric",
                y="Metric Value",
                color="Data Type",
            )
        )
        plot.save(
            cons.PATHS.DATA_REPORTING / "classification_metrics_plot.png",
            format="png",
            bbox_inches="tight",
        )

    def _save_violin_plot(self, error_table: pd.DataFrame):
        """
        Builds and saves a plot showing error distributions for each attribute and each data type approach.

        Parameters
        ----------
        error_table
            Dataframe with error values for a specific attribute, e.g. relative error values for volume, faces, edges and vertices.

        Returns
        -------
        None
        """
        logger.info("Saving violin plot for error distributions")
        # filter the error table to only include relative errors
        error_table_relative_errors = error_table.query(
            "error_type.str.contains('relative')"
        ).copy()
        ax = sns.violinplot(
            data=error_table_relative_errors,
            x="data_type",
            y="value",
            hue="error_type",
        )
        logger.debug(type(ax))
        ax.set_title("Error Distributions of Regression Tasks")
        ax.set_xlabel("Data Representation Type")
        ax.set_ylabel("Relative Error")
        # Set custom legend labels
        legend_labels = ["Volume", "# of Faces", "# of Edges", "# of Vertices"]
        legend = ax.get_legend()
        legend.set_title("Regression Values")
        for t, l in zip(legend.texts, legend_labels):
            t.set_text(l)

        ax.figure.set_size_inches(15, 8)
        ax.figure.savefig(
            cons.PATHS.DATA_REPORTING / "error_distributions.png",
            format="png",
            bbox_inches="tight",
        )

    def _save_confusion_matrix_plot(self, confusion_matrix: pd.DataFrame, data_type: str) -> None:
        """
        """
        ax = sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            cbar=False,
            xticklabels=confusion_matrix.columns,
            yticklabels=confusion_matrix.index,
        )
        ax.set_title(f"Confusion Matrix for {data_type}")
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("True Class")
        ax.figure.set_size_inches(15, 8)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")

        ax.figure.savefig(
            cons.PATHS.DATA_REPORTING / f"confusion_matrix_{data_type}.png",
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
        try:
            classifier_output = pd.read_csv(
                cons.PATHS.DATA_MODEL_OUTPUT / "classifiers_output.csv"
            )
            test_data = pd.read_csv(cons.PATHS.DATA_MODEL_INPUT / "test.csv")[
                ["path", "class_id", "class_name"]
            ]
            classifier_output_found = True
        except FileNotFoundError:
            classifier_output_found = False

        if classifier_output_found:
            # calculate and save confusion matrices for each data type
            for data_type in classifier_output["data_type"].unique():
                logger.info(f"Processing data type: {data_type}")
                class_ids_true, class_ids_predicted = self._get_true_and_prediced_values(
                    classifier_output, test_data, data_type, is_classifier=True
                )
                class_id_to_class_name = test_data.drop_duplicates(subset=["class_id"])[["class_id", "class_name"]].set_index("class_id")["class_name"].to_dict()
                confusion_matrix = self._get_confusion_matrix(
                    class_ids_true, class_ids_predicted, data_type, class_id_to_class_name
                )
                self._save_confusion_matrix_plot(confusion_matrix, data_type)
        

            # compute classification metrics and plot them for all models
            classification_metrics = self._get_classification_metrics(classifier_output, test_data)
            self._save_classification_metrics_plot(classification_metrics)

        # === REGRESSION METRICS ===
        logger.info("Compute regression metrics.")
        # load predictions and test data
        try:
            regressor_output = pd.read_csv(
                cons.PATHS.DATA_MODEL_OUTPUT / "regressors_output.csv"
            )
            test_data = pd.read_csv(cons.PATHS.DATA_MODEL_INPUT / "test.csv")[
                ["path", "volume", "faces", "edges", "vertices"]
            ]
            regressor_output_found = True
        except FileNotFoundError:
            regressor_output_found = False

        if regressor_output_found:
            # save regression metrics as csv table and plot
            regression_metrics = self._get_regression_metrics(regressor_output, test_data)
            regression_metrics.to_csv(cons.PATHS.DATA_REPORTING / "regression_metrics_table.csv", index=False)
            self._save_regression_metrics_plot(regression_metrics)

            # save error distributions as violin plot
            error_table = self._get_error_table(regressor_output, test_data)
            self._save_violin_plot(error_table)

        
        logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    pipeline = ModelOutputToReportingPipeline()
    pipeline.run()
# %%
