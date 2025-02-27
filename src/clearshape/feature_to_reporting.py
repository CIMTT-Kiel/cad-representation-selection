"""
Pipeline to analyze the `4_feature` data set and create a report on the data set.

The analysis includes:
# TODO complete this list
"""

# standard libary
import logging

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

class FeatureReportingPipeline:

    def __init__(self):
        pass

    def _load_data(self):
        """
        Set `targets` attribute to the data in `fabwave_targets.csv`.
        
        The data is converted to long format.
        """
        self.targets = pd.read_csv(cons.PATHS.DATA_FEATURE / "fabwave_targets.csv")
        self.targets = self.targets.melt(value_vars=self.targets.columns, var_name="target", value_name="value", ignore_index=False)

    def _save_descriptive_statistics(self):
        fabwave_targets = pd.read_csv(cons.PATHS.DATA_FEATURE / "fabwave_targets.csv")
        description = fabwave_targets.describe(include="all")
        description.to_csv(cons.PATHS.DATA_REPORTING / "feature/descriptive_statistics.csv")

    def _plot_class_distribution(self):
        figure, axis = plt.subplots(1, 1, figsize=(10, 6))
        plot = (
            so.Plot(self.targets.query("target=='class_name'"), x="value")
            .add(so.Bar(), so.Hist())
            .on(axis)
            )
        axis.tick_params(axis="x", rotation=90)
        figure.tight_layout()
        plot.save(cons.PATHS.DATA_REPORTING / "feature/class_distribution.png")

    def _plot_regression_target_distributions(self):
        figure, axes = plt.subplots(4, 1, figsize=(6, 12))
        for i, target in enumerate(["volume","faces", "edges", "vertices"]):
            (
                so.Plot(self.targets.query(f"target == '{target}'"), x="value")
                .add(so.Area(),so.Hist(bins=50))
                .on(axes[i])
                .plot()
            )
            axes[i].set_xlabel(target)
        figure.tight_layout()
        figure.suptitle("Regression Target Distributions")
        figure.savefig(cons.PATHS.DATA_REPORTING / "feature/regression-target-dist.png")

    def run(self):
        self._load_data()
        self._save_descriptive_statistics()
        #self._plot_class_distribution()
        #self._plot_regression_target_distributions()

if __name__ == "__main__":
    pipeline = FeatureReportingPipeline()
    pipeline.run()