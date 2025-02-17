"""
Pipeline to convert 'primary' data to 'feature' data.

The `.step` files from the 'primary' data set are converted into the three representaiton formats: 'images', 'invariants' and 'trees'.
In each case case the representations are stored in a folder named after the representation type. The folder is located in the 'feature' folder.
Also, as part of the feature generation process, a table with regression featuers is created. For each CAD-part it contains the following features:

- Volume
- Amount of Faces
- Amount of Edges
- Amount of Vertices

Notes
-----
The excecution of the pipeline may be interrupted and resumed at any time. The pipeline will skip files that have already been processed.
"""

# standard libary imports
import logging
from pathlib import Path
from tqdm import tqdm
import time
import threading

# third party imports
import pandas as pd
from omegaconf import OmegaConf
import cadquery as cq
import dgl

# custom imports
import clearshape.constants as cons
from clearshape.step_tree.step_tree import StepTree

# set up logger
logging_level = logging.DEBUG
logger = logging.getLogger(__name__)
logger.setLevel(logging_level)
formatter = logging.Formatter("%(asctime)s %(levelname)8s - %(message)s")
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging_level)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


class PrimaryFeaturePipeline:
    """
    A pipeline for processing CAD models and extracting features.

    This class implements a singleton pattern to ensure only one instance of the pipeline exists.
    It processes STEP files from a primary data set, converts them to various representations,
    extracts regression features and part classes and saves the processed data.

    Parameters
    ----------
    None

    Attributes
    ----------
    _instance : PrimaryFeaturePipeline
        The singleton instance of the class.
    _conf : OmegaConf
        Configuration loaded from a YAML file.
    _step_path_generator : generator
        Generator for iterating over STEP files in the primary data set.
    _targets : list
        List to store extracted regression features.
    _known_classes : list
        List to store known class names.
    _file_to_process : Path
        Path to the current STEP file being processed.
    _step_tree : DGLGraph
        Tree representation of the current STEP file.
    """

    _instance = None

    def __new__(cls):
        """
        Create a new instance of the class if one does not already exist.

        This method ensures that only one instance of the pipeline class can exist
        (Singleton pattern). If an instance already exists, it returns the existing
        instance. Otherwise, it creates a new instance and returns it.

        Returns:
            cls: The single instance of the class.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self._conf = OmegaConf.load(
            cons.PATHS.CONFIG / "primary_to_feature_pipeline.yaml"
        )
        self._step_path_generator = (
            cons.PATHS.DATA_PRIMARY / "fabwave"
        ).rglob("*.step")
        self._targets= []
        self._known_classes = []

        # get list of files already processed
        try:
            self._files_already_processed = pd.read_csv(cons.PATHS.DATA_FEATURE / "fabwave_targets.csv")["path"].values
        except pd.errors.EmptyDataError:
            self._files_already_processed = None
        except FileNotFoundError:
            self._files_already_processed = None

    def _get_next_step_path(self) -> None:
        """
        Retrieve the path to the next step file from the primary data set.

        This method uses a step path generator to obtain the path to the next file
        that needs to be processed. It updates the instance variable `_file_to_process`
        with the path to this file.

        Returns
        -------
        None

        Raises
        ------
        StopIteration: If the step path generator has no more files to process.
        """
        # get path to next step file
        logger.debug("Getting next step file")
        self._file_to_process = next(self._step_path_generator)

    def _get_targets(self) -> None:
        """
        Extract regression features from the CAD part and the classification
        target from the file path.

        This method performs the following steps:
        1. Loads the CAD part from the file path using the `cq.importers.importStep` method.
        2. Determines the class name from the parent directory of the file path.
        3. Adds the class name to the list of known classes if it is not already present.
        4. Retrieves the class ID based on the index of the class name in the list of known classes.
        5. Extracts regression features from the CAD part, including volume, number of faces, edges, and vertices.
        6. Constructs the relative path of the part. (without the file extension)
        7. Appends a dictionary containing the part path, class name, class ID, and extracted features to the `_targets` list.

        Returns:
            None
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
        Convert the STEP file to a tree representation as a DGL graph.

        This method reads a STEP file, converts it into a tree structure using the StepTree class,
        and then transforms that tree into a DGL (Deep Graph Library) graph. The resulting graph
        is stored in the instance variable `_step_tree`.
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
        #return NotImplemented
        print("runing image conversion")

    # TODO: Implement invariant conversion
    def _convert_to_invariants(self):
        """
        # TODO: Add method docstring
        """
        # self._invariants =
        #return NotImplemented
        print("runing invariants conversion")

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

    def _save_targets(self):
        """
        Save the extracted regression features.
        """
        logger.debug("Saving targets")
        if self._files_already_processed is not None:
            targets_already_processed = pd.read_csv(cons.PATHS.DATA_FEATURE / "fabwave_targets.csv")
        else:
            targets_already_processed = None
        new_targets = pd.DataFrame(self._targets)
        targets_all = pd.concat([targets_already_processed, new_targets], ignore_index=True)
        targets_all.to_csv(
            cons.PATHS.DATA_FEATURE / "fabwave_targets.csv", index=False
        )

    def _save_class_names(self):
        """
        Save the class names with their corresponding ids.
        """
        logger.debug("Saving class names")
        targets = pd.read_csv(cons.PATHS.DATA_FEATURE / "fabwave_targets.csv")
        class_names = targets[["class_name", "class_id"]].drop_duplicates()
        class_names.to_csv(
            cons.PATHS.DATA_REPORTING / "part_class_ids.csv", index=False
        )

    def run(self):
        """
        Execute the entire pipeline.
        """

        # process all step files
        logger.info("Starting processing of step files.")
        progress_bar = tqdm(desc="Files processed: ",total=len(list(Path(cons.PATHS.DATA_PRIMARY / "fabwave").rglob("*.step"))), ncols=100)

        try:
            while True:
                try:
                    self._get_next_step_path()
                    # skip if file already processed 
                    relative_path = self._file_to_process.relative_to(
                        cons.PATHS.DATA_PRIMARY / "fabwave"
                    ).with_suffix("").as_posix()
                    if self._files_already_processed is not None and relative_path in self._files_already_processed:
                        logger.debug(f"Skipping already processed file {self._file_to_process}")
                        progress_bar.update(1)
                        continue

                    logger.debug(f"Processing {self._file_to_process}")
                except StopIteration:
                    logger.info("All step files processed")
                    break

                try:
                    tree_thread = threading.Thread(target=self._convert_to_tree)
                    # TODO: Implement image and invariant conversion
                    # self._convert_to_image()
                    # self._convert_to_invariant()
                    image_thread = threading.Thread(target=self._convert_to_image)
                    invariant_thread = threading.Thread(target=self._convert_to_invariants)

                    logger.debug("Starting threads")
                    tree_thread.start()
                    image_thread.start()
                    invariant_thread.start()

                    tree_thread.join()
                    image_thread.join()
                    invariant_thread.join()

                    logger.debug("Threads finished")


                    # Save all data representations only if all conversions are successful
                    self._save_data()
                    
                    self._get_targets()  # Extract regression features and class labels

                    progress_bar.update(1)

                except Exception as e:
                    logger.warning(f"Error processing {self._file_to_process}: {e}")
                    continue
        finally:
            self._save_targets()
            self._save_class_names()

if __name__ == "__main__":
    pipeline = PrimaryFeaturePipeline()
    pipeline.run()
