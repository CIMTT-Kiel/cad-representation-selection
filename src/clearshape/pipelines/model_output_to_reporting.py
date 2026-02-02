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
import matplotlib.pyplot as plt

# custom packages
import clearshape.constants as cons

# set up logger
logging_level = logging.INFO  # <== set logging level here
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
        logger.debug(f"Entered _get_true_and_prediced_values with arguments:")
        logger.debug(f"\tdata_type: {data_type}")
        logger.debug(f"\tis_classifier: {is_classifier}")
        data_subset = output.query("data_type == @data_type")
        # assert all(data_subset["path"] == test_data["path"])

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
        logger.debug(f"Entered _get_confusion_matrix")
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
        logger.debug(f"Entered _get_classification_metrics")
        results = []
        for data_type in classifier_output["data_type"].unique():
            class_ids_true, class_ids_predicted = self._get_true_and_prediced_values(
                classifier_output, test_data, data_type, is_classifier=True
            )
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
        logger.debug("Entered _get_regression_metrics")
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
        logger.debug("Entered _save_regression_metrics_plot")

        # Create separate plots for each metric type to handle different scales
        for metric_type in ['mae', 'r2', 'mse']:
            metric_data = regression_metrics[regression_metrics['metric'] == metric_type].copy()

            # Capitalize data_type for better visualization
            metric_data['data_type'] = metric_data['data_type'].str.capitalize()
            metric_data['attribute'] = metric_data['attribute'].str.capitalize()

            # Set up the plot
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            fig.suptitle(f'Regression Metrics: {metric_type.upper()}', fontsize=14, fontweight='bold')

            attributes = ['Volume', 'Faces', 'Edges', 'Vertices']
            for idx, attribute in enumerate(attributes):
                attr_data = metric_data[metric_data['attribute'] == attribute]

                if not attr_data.empty:
                    ax = axes[idx]
                    # Create bar plot
                    data_types = attr_data['data_type'].unique()
                    x_pos = range(len(data_types))
                    values = [attr_data[attr_data['data_type'] == dt]['value'].values[0] for dt in data_types]

                    bars = ax.bar(x_pos, values, color=['#4C72B0', '#DD8452'])
                    ax.set_xlabel('Data Type')
                    ax.set_ylabel(metric_type.upper())
                    ax.set_title(attribute)
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(data_types, rotation=45, ha='right')

                    # Add value labels on bars
                    for bar, value in zip(bars, values):
                        height = bar.get_height()
                        if metric_type == 'r2':
                            label = f'{value:.3f}'
                        elif metric_type == 'mse' and value > 1000:
                            label = f'{value:.2e}'
                        else:
                            label = f'{value:.1f}'
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               label, ha='center', va='bottom', fontsize=9)

            plt.tight_layout()
            plt.savefig(
                cons.PATHS.DATA_REPORTING / f"regression_metrics_{metric_type}_plot.png",
                format="png",
                bbox_inches="tight",
                dpi=150
            )
            plt.close(fig)

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
        logger.debug("Entered _get_error_table")
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

    def _save_classification_metrics_plot(
        self, classification_metrics: pd.DataFrame
    ) -> None:
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
        logger.debug("Entered _save_classification_metrics_plot")
        # Overwrite string values for proper visualization
        classification_metrics["data_type"] = classification_metrics[
            "data_type"
        ].str.capitalize()
        for metric, display_name in zip(
            ["accuracy", "f1_score", "recall", "precision"],
            ["Accuracy", "F1", "Recall", "Precision"],
        ):
            classification_metrics.loc[
                classification_metrics["metric"] == f"{metric}_macro", "metric"
            ] = display_name

        plot = (
            so.Plot(classification_metrics, x="metric", y="value", color="data_type")
            .add(so.Bar(), so.Dodge())
            .label(
                title="Classification Metrics (Macro Averaged)",
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
        logger.debug("Entered _save_violin_plot")
        # filter the error table to only include relative errors
        error_table_relative_errors = error_table.query(
            "error_type.str.contains('relative')"
        ).copy()

        # Capitalize data types for better visualization
        error_table_relative_errors['data_type'] = error_table_relative_errors['data_type'].str.capitalize()

        # Create figure with subplots for each attribute
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Error Distributions of Regression Tasks", fontsize=16, fontweight='bold')

        attributes = [
            ('volume_relative_error', 'Volume', axes[0, 0]),
            ('faces_relative_error', '# of Faces', axes[0, 1]),
            ('edges_relative_error', '# of Edges', axes[1, 0]),
            ('vertices_relative_error', '# of Vertices', axes[1, 1])
        ]

        for error_type, label, ax in attributes:
            attr_data = error_table_relative_errors[
                error_table_relative_errors['error_type'] == error_type
            ].copy()

            if not attr_data.empty:
                # Calculate quantiles to handle outliers
                q95 = attr_data['value'].quantile(0.95)
                q99 = attr_data['value'].quantile(0.99)

                # Create violin plot
                sns.violinplot(
                    data=attr_data,
                    x="data_type",
                    y="value",
                    ax=ax,
                    cut=0,
                    inner="box",
                )

                # Set y-limit to 95th percentile for better visualization
                ax.set_ylim(0, max(q95 * 1.1, 0.1))

                ax.set_title(f'{label}\n(showing up to 95th percentile)', fontsize=12)
                ax.set_xlabel('Data Representation Type', fontsize=10)
                ax.set_ylabel('Relative Error', fontsize=10)

                # Add statistics text
                for i, data_type in enumerate(attr_data['data_type'].unique()):
                    dt_data = attr_data[attr_data['data_type'] == data_type]['value']
                    median = dt_data.median()
                    mean = dt_data.mean()
                    outliers = (dt_data > q95).sum()
                    total = len(dt_data)

                    stats_text = f'Med: {median:.2f}\nMean: {mean:.2f}\nOutliers: {outliers}/{total}'
                    ax.text(i, ax.get_ylim()[1] * 0.7, stats_text,
                           ha='center', va='center', fontsize=8,
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(
            cons.PATHS.DATA_REPORTING / "error_distributions.png",
            format="png",
            bbox_inches="tight",
            dpi=150
        )
        plt.close(fig)

    def _save_prediction_scatter_plots(self, regressor_output: pd.DataFrame, test_data: pd.DataFrame):
        """
        Creates scatter plots comparing predicted vs actual values for each attribute and data type.

        Parameters
        ----------
        regressor_output : pd.DataFrame
            DataFrame containing the regressor output.
        test_data : pd.DataFrame
            DataFrame containing the test data with true values.

        Returns
        -------
        None
        """
        logger.debug("Entered _save_prediction_scatter_plots")

        attributes = ['volume', 'faces', 'edges', 'vertices']
        fig, axes = plt.subplots(len(attributes), 2, figsize=(14, 16))
        fig.suptitle('Predicted vs Actual Values', fontsize=16, fontweight='bold')

        for row_idx, attribute in enumerate(attributes):
            for col_idx, data_type in enumerate(regressor_output['data_type'].unique()):
                ax = axes[row_idx, col_idx]

                # Get data for this combination
                true_values, predicted_values = self._get_true_and_prediced_values(
                    regressor_output, test_data, data_type, is_classifier=False
                )

                y_true = true_values[attribute].values
                y_pred = predicted_values[f'pred_{attribute}'].values

                # Create scatter plot
                ax.scatter(y_true, y_pred, alpha=0.5, s=20)

                # Add perfect prediction line
                min_val = min(y_true.min(), y_pred.min())
                max_val = max(y_true.max(), y_pred.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')

                # Calculate R²
                from sklearn.metrics import r2_score
                r2 = r2_score(y_true, y_pred)

                ax.set_xlabel(f'Actual {attribute.capitalize()}', fontsize=10)
                ax.set_ylabel(f'Predicted {attribute.capitalize()}', fontsize=10)
                ax.set_title(f'{data_type.capitalize()} - {attribute.capitalize()}\nR² = {r2:.3f}', fontsize=11)
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            cons.PATHS.DATA_REPORTING / "prediction_vs_actual_scatter.png",
            format="png",
            bbox_inches="tight",
            dpi=150
        )
        plt.close(fig)

    def _save_model_comparison_summary(self, classification_metrics: pd.DataFrame, regression_metrics: pd.DataFrame):
        """
        Creates a summary plot comparing all models across classification and regression tasks.

        Parameters
        ----------
        classification_metrics : pd.DataFrame
            DataFrame with classification metrics.
        regression_metrics : pd.DataFrame
            DataFrame with regression metrics.

        Returns
        -------
        None
        """
        logger.debug("Entered _save_model_comparison_summary")

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # Classification metrics comparison
        ax1 = fig.add_subplot(gs[0, :])
        class_data = classification_metrics.copy()
        class_data['data_type'] = class_data['data_type'].str.capitalize()

        # Pivot for grouped bar chart
        class_pivot = class_data.pivot(index='metric', columns='data_type', values='value')
        class_pivot.plot(kind='bar', ax=ax1, rot=0, color=['#4C72B0', '#DD8452'])
        ax1.set_title('Classification Metrics Comparison', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Metric', fontsize=11)
        ax1.set_ylabel('Score', fontsize=11)
        ax1.set_ylim(0, 1)
        ax1.legend(title='Data Type', fontsize=10)
        ax1.grid(axis='y', alpha=0.3)

        # Regression R² comparison
        ax2 = fig.add_subplot(gs[1, 0])
        r2_data = regression_metrics[regression_metrics['metric'] == 'r2'].copy()
        r2_data['data_type'] = r2_data['data_type'].str.capitalize()
        r2_data['attribute'] = r2_data['attribute'].str.capitalize()

        r2_pivot = r2_data.pivot(index='attribute', columns='data_type', values='value')
        r2_pivot.plot(kind='bar', ax=ax2, rot=45, color=['#4C72B0', '#DD8452'])
        ax2.set_title('R² Score Comparison (Regression)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Attribute', fontsize=10)
        ax2.set_ylabel('R² Score', fontsize=10)
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax2.legend(title='Data Type', fontsize=9)
        ax2.grid(axis='y', alpha=0.3)

        # Regression MAE comparison - excluding Volume for better scale
        ax3 = fig.add_subplot(gs[1, 1])
        mae_data = regression_metrics[regression_metrics['metric'] == 'mae'].copy()
        mae_data['data_type'] = mae_data['data_type'].str.capitalize()
        mae_data['attribute'] = mae_data['attribute'].str.capitalize()

        # Split Volume from other attributes for separate visualization
        mae_data_no_volume = mae_data[mae_data['attribute'] != 'Volume']
        mae_data_volume = mae_data[mae_data['attribute'] == 'Volume']

        # Plot non-volume attributes on primary y-axis
        mae_pivot_no_volume = mae_data_no_volume.pivot(index='attribute', columns='data_type', values='value')
        mae_pivot_no_volume.plot(kind='bar', ax=ax3, rot=45, color=['#4C72B0', '#DD8452'], width=0.7, position=1)
        ax3.set_ylabel('MAE (Faces, Edges, Vertices)', fontsize=10, color='black')
        ax3.tick_params(axis='y', labelcolor='black')
        ax3.grid(axis='y', alpha=0.3)

        # Create secondary y-axis for Volume
        ax3_twin = ax3.twinx()

        # Plot Volume on secondary y-axis
        mae_pivot_volume = mae_data_volume.pivot(index='attribute', columns='data_type', values='value')
        x_pos = len(mae_pivot_no_volume)  # Position after other attributes
        width = 0.35
        data_types = ['Vecsets', 'Invariants']
        colors = ['#4C72B0', '#DD8452']

        for i, dtype in enumerate(data_types):
            if dtype in mae_pivot_volume.columns:
                value = mae_pivot_volume[dtype].values[0]
                ax3_twin.bar(x_pos + i * width - width/2, value, width,
                           color=colors[i], alpha=0.8, label=dtype if x_pos == len(mae_pivot_no_volume) else "")

        ax3_twin.set_ylabel('MAE (Volume)', fontsize=10, color='#8B4513')
        ax3_twin.tick_params(axis='y', labelcolor='#8B4513')

        # Update x-axis to include Volume
        all_attributes = list(mae_pivot_no_volume.index) + ['Volume']
        ax3.set_xticks(range(len(all_attributes)))
        ax3.set_xticklabels(all_attributes, rotation=45, ha='right')

        ax3.set_title('MAE Comparison (Regression)\n(Volume on right axis)', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Attribute', fontsize=10)

        # Combine legends
        lines1, labels1 = ax3.get_legend_handles_labels()
        ax3.legend(lines1, labels1, title='Data Type', fontsize=9, loc='upper left')

        plt.savefig(
            cons.PATHS.DATA_REPORTING / "model_comparison_summary.png",
            format="png",
            bbox_inches="tight",
            dpi=150
        )
        plt.close(fig)

    def _save_confusion_matrix_plot(self, confusion_matrix: pd.DataFrame, data_type: str) -> None:
        """
        """
        logger.debug("Entered _save_confusion_matrix_plot")
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
        plt.close(ax.figure)

    def run(self):
        """
        Execute the pipeline to compute reporting metrics and save confusion matrices.

        Returns
        -------
        None
        """
        logger.info("Starting Model Output to Reporting Pipeline...")

        # === CLASSIFICATION METRICS ===
        logger.info("=== COMPUTE CLASSIFIER METRICS ===")
        # load predictions and test data
        try:
            logger.info("Loading classifier output and test data.")
            classifier_output = pd.read_csv(
                cons.PATHS.DATA_MODEL_OUTPUT / "classifiers_output.csv"
            )
            test_data = pd.read_csv(cons.PATHS.DATA_MODEL_INPUT / "test.csv")[
                ["path", "class_id", "class_name"]
            ]
            classifier_output_found = True
        except FileNotFoundError:
            logger.warning("No classifier output found. Skipping classification metrics.")
            classifier_output_found = False

        if classifier_output_found:
            # calculate and save confusion matrices for each data type
            for data_type in classifier_output["data_type"].unique():
                logger.info(f"Calculating and saving confusion matrix for: {data_type}")
                class_ids_true, class_ids_predicted = self._get_true_and_prediced_values(
                    classifier_output, test_data, data_type, is_classifier=True
                )
                class_id_to_class_name = test_data.drop_duplicates(subset=["class_id"])[["class_id", "class_name"]].set_index("class_id")["class_name"].to_dict()
                confusion_matrix = self._get_confusion_matrix(
                    class_ids_true, class_ids_predicted, data_type, class_id_to_class_name
                )
                self._save_confusion_matrix_plot(confusion_matrix, data_type)

            # compute classification metrics and plot them for all models
            logger.info("Calculating and saving classification metrics.")
            classification_metrics = self._get_classification_metrics(classifier_output, test_data)
            classification_metrics.to_csv(
                cons.PATHS.DATA_REPORTING / "classification_metrics_table.csv", index=False
            )
            self._save_classification_metrics_plot(classification_metrics)

        # === REGRESSION METRICS ===
        logger.info("=== COMPUTE REGRESSION METRICS ===")
        # load predictions and test data
        try:
            logger.info("Loading regressor output and test data.")
            regressor_output = pd.read_csv(
                cons.PATHS.DATA_MODEL_OUTPUT / "regressors_output.csv"
            )
            test_data = pd.read_csv(cons.PATHS.DATA_MODEL_INPUT / "test.csv")[
                ["path", "volume", "faces", "edges", "vertices"]
            ]
            regressor_output_found = True
        except FileNotFoundError:
            logger.warning("No regressor output found. Skipping regression metrics.")
            regressor_output_found = False

        if regressor_output_found:
            # save regression metrics as csv table and plot
            logger.info("Calculating and saving regression metrics.")
            regression_metrics = self._get_regression_metrics(regressor_output, test_data)
            regression_metrics.to_csv(cons.PATHS.DATA_REPORTING / "regression_metrics_table.csv", index=False)
            self._save_regression_metrics_plot(regression_metrics)

            # save error distributions as violin plot
            logger.info("Calculating and saving error distributions. (Violin Plot)")
            error_table = self._get_error_table(regressor_output, test_data)
            self._save_violin_plot(error_table)

            # save prediction vs actual scatter plots
            logger.info("Creating prediction vs actual scatter plots.")
            self._save_prediction_scatter_plots(regressor_output, test_data)

        # Create combined summary plot if both metrics are available
        if classifier_output_found and regressor_output_found:
            logger.info("Creating model comparison summary.")
            self._save_model_comparison_summary(classification_metrics, regression_metrics)

        logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    pipeline = ModelOutputToReportingPipeline()
    pipeline.run()
# %%
