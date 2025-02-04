"""
Pipeline to convert 'primary' data to 'feature' data.

The `.step` files from the 'primary' data set are converted into the three representaiton formats: 'images', 'invariants' and 'trees'.
In each case case the representations are stored in a folder named after the representation type. The folder is located in the 'feature' folder.
Also, as part of the feature generation process, a table with regression featuers is created. For each CAD-part it contains the following features:

- Volume
- Amount of Faces
- Amount of Edges
- Amount of Vertices
"""

# IDEA
# 
# If a part shows up in the regression features table it has been successfully
# converted into a all three representations. Thus that part can be skipped when
# excecuting the pipeline again. But only if the conversion logic for any of the
# representations has not changed.


# standard libary imports
import logging
from pathlib import Path

# third party imports
import pandas as pd
from omegaconf import OmegaConf
import cadquery as cq
import dgl

# custom imports
import clearshape.constants as cons
from clearshape.step_tree.step_tree import StepTree

# set up logger
logging_level = logging.INFO
logger = logging.getLogger(__name__)
logger.setLevel(logging_level)
formatter = logging.Formatter("%(levelname)s %(asctime)s - %(message)s")
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging_level)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


class PrimaryFeaturePipeline:
    """
    # TODO: Add class docstring
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
        """
        # TODO: Add method docstring
        """
        self._conf = OmegaConf.load(
            cons.PATHS.CONFIG / "primary_to_feature_pipeline.yaml"
        )
        # TODO remove "Gears". Just for testing
        self._step_path_generator = (
            cons.PATHS.DATA_PRIMARY / "fabwave/Pipe_Fittings"
        ).rglob("*.step")
        self._targets= []
        self._known_classes = []

    def _get_next_step_path(self):
        """
        Get the next step file from the primary data set.
        """
        # get path to next step file
        logger.debug("Getting next step file")
        self._file_to_process = next(self._step_path_generator)

    def _get_targets(self):
        """
        Extract regression features from the CAD part and the classifiction
        target. from the file path.
        """
        logger.debug("Extracting regression features")
        # load the CAD part
        part = cq.importers.importStep(self._file_to_process.as_posix())
        # class target
        class_name = self._file_to_process.parent.name
        if class_name not in self._known_classes:
            self._known_classes.append(class_name)
        class_id = self._known_classes.index(class_name)
        # extract regression features
        volume = part.val().Volume()
        faces = len(part.val().Faces())
        edges = len(part.val().Edges())
        vertices = len(part.val().Vertices())

        part_path = self._file_to_process.relative_to(
            cons.PATHS.DATA_PRIMARY / "fabwave"
        ).with_suffix("").as_posix()

        self._targets.append(
            {
                "path": part_path,
                "class_name": class_name,
                "class_id": class_id,
                "volume": volume,
                "faces": faces,
                "edges": edges,
                "vertices": vertices,
            }
        )

    def _convert_to_tree(self):
        """
        Convert the step file to a tree representation.
        """
        logger.debug("Converting CAD model to tree representation")
        # create a DGL graph from the step file
        self._step_tree = StepTree.from_step_file(self._file_to_process).to_dgl_graph()

    # TODO: Implement image conversion
    def _convert_to_image(self):
        """
        # TODO: Add method docstring
        """
        # self._images =
        return NotImplemented

    # TODO: Implement invariant conversion
    def _convert_to_invariants(self):
        """
        # TODO: Add method docstring
        """
        # self._invariants =
        return NotImplemented

    # TODO add saving code for images and invariants
    def _save_data(self):
        """
        Save the processed data and extracted features.
        """
        logger.debug("Saving data")
        # save tree representation
        relative_path = self._file_to_process.relative_to(
            cons.PATHS.DATA_PRIMARY
        ).with_suffix(".bin")
        tree_path = (cons.PATHS.DATA_FEATURE / "trees" / relative_path).as_posix()
        dgl.save_graphs(tree_path, [self._step_tree])

        # save invariants
        # self._invariants.to_json()

        # save images
        # self._images.save()

    def run(self):
        """
        Execute the entire pipeline.
        """
        while True:
            try:
                self._get_next_step_path()
            except StopIteration:
                logger.info("All step files processed")
                break

            try:
                # Convert to tree representation
                self._convert_to_tree()

                # TODO: Implement image and invariant conversion
                # self._convert_to_image()
                # self._convert_to_invariant()

                # Extract regression features
                self._get_targets()

                # Save all data representations only if all conversions are successful
                self._save_data()

            except Exception as e:
                logger.error(f"Error processing {self._file_to_process}: {e}")
                continue

        regression_features = pd.DataFrame(self._regression_features)
        regression_features.to_csv(
            cons.PATHS.DATA_FEATURE / "regression_features_fabwave.csv", index=False
        )


if __name__ == "__main__":
    pipeline = PrimaryFeaturePipeline()
    pipeline.run()
