"""
Pipeline to convert 'primary' data to 'feature' data.

See `PrimaryFeaturePipeline` for more details.
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
import dgl
import numpy as np

# custom imports
from clearshape import constants as cons
from clearshape.step_tree.step_tree import StepTree
from clearshape.invariants.invariant import InvariantCalculator
from clearshape.targets.STEP_targets import RegressionTargetExtractor
from clearshape.vecsets.preprocessing.conversions import CAD_Converter

# set up logger
logging_level = logging.DEBUG
logger = logging.getLogger(__name__)
logger.setLevel(logging_level)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)8s - %(message)s")
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging_level)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


class PrimaryFeaturePipeline:
    """
    A pipeline for processing CAD models and extracting features.

    The images are generated externally and manually stored in the `data/4_feature/images` folder.

    The `.step` files from the 'primary' data set are converted into 'invariants' and 'trees' representations.
    Each representation is stored if it is successfully created. Is not guaranteed that all representations are created for each step file.

    In each case case the representations are stored in a folder named after the representation type. The folder is located in the 'feature' folder.

    Also, as part of the feature generation process, a table with regression and classification featuers is created. Each row of the table corresponds to a CAD part for which all three representations are available.
    The table is stored in the `data/4_feature/fabwave_targets.csv` file. It contains the following columns:

    - path, relative path to the part, without the file extension
    - class_name, name of the class the part belongs to
    - class_id, id of the class the part belongs to
    - volume, volume of the part in mm^3
    - faces, number of faces of the part
    - edges, number of edges of the part
    - vertices, number of vertices of the part

    This 'fabwave_targets.csv' file contains only entries for the parts for which all three representations are available.
    Thus it serves as the central reference for the feature_to_model_input pipeline.

    The excecution of the pipeline may be interrupted and resumed at any time. The pipeline will skip files that have already been processed. (This is determined by the contents of `fabwave_targets.csv` file.)

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

    Methods
    -------
    run()
        Execute the entire pipeline, processing all STEP files and extracting features.

    Notes
    -----
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
        try:
            self._known_classes = pd.read_csv(cons.PATHS.DATA_FEATURE / "fabwave_targets.csv")["class_name"].unique().tolist()
        except (pd.errors.EmptyDataError, FileNotFoundError):
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
        logger.info("Setting next step file")
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
        logger.debug("Extracting regression and classification features")

        class_name = self._file_to_process.parent.name
        if class_name not in self._known_classes:
            self._known_classes.append(class_name)
        class_id = self._known_classes.index(class_name)

        # extract regression features
        extractor, targets = RegressionTargetExtractor.analyze_step(self._file_to_process)

        volume = targets["volume"]
        faces = targets["faces"]
        edges = targets["edges"]
        vertices = targets["vertices"]

        # not used for now
        #dx = targets["dx"]
        #dy = targets["dy"]
        #dz = targets["dz"]

        # delete extractor and targets to free up memory
        del extractor
        del targets

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

    def _convert_to_tree(self) -> None:
        """
        Convert the STEP file to a tree representation as a DGL graph.

        This method reads a STEP file, converts it into a tree structure using the StepTree class,
        and then transforms that tree into a DGL (Deep Graph Library) graph. The resulting graph
        is stored in the instance variable `_step_tree`.
        """
        logger.debug("Converting CAD model to tree representation")
        # create a DGL graph from the step file
        self._step_tree = StepTree.from_step_file(self._file_to_process).to_dgl_graph()

    def _convert_to_invariants(self) -> None:
        """
        Calculate invariants based on a STEP file.

        This method reads a STEP file, converts the file to STL as an interim state and then try to mesh the geometry with tetraedons. Based on the meshed geometry the invariants are calculated and is stored in the instance variables _mues for the stat. moments and _pis for the invariants both as dictionaries with the permutations of p,q and r as keys in the form 'mue_{p}{q}{r}' and 'pi_{p}{q}{r}'. The permutations used for the calculation are stored in the class variable moment_permutations.  
        """
        logger.debug("Calculate the moments and invariants from the CAD model")
        self._invariants = InvariantCalculator.calculate_invariants_from_step(self._file_to_process)
        #return NotImplemented
        logger.info("running invariants conversion")

    def _convert_to_vecset(self) -> None:
        """
        Convert the STEP file to a vector set representation.

        This method reads a STEP file and converts it into a vector set representation.
        The vector set is stored in the instance variable `_vecset`.
        """
        logger.debug("Converting CAD model to vector set representation")

        self._vecset = CAD_Converter(self._file_to_process).to_vecset()

    def _get_relative_path(self) -> Path:
        """
        Get the relative path of the current file to process.

        Returns
        -------
        Path
            The relative path of the current file to process.
        """
        relative_path = self._file_to_process.relative_to(
            cons.PATHS.DATA_PRIMARY
        ).with_suffix(".bin")
        return relative_path

    def _save_tree(self) -> None:
        relative_path = self._get_relative_path()
        tree_path = (cons.PATHS.DATA_FEATURE / "trees" / relative_path).as_posix()
        dgl.save_graphs(tree_path, [self._step_tree])

    def _tree_available(self) -> bool:
        """
        Check if the tree representation is available for the current part.

        This method checks if a DGL graph file exists for the current part,
        specified by `_file_to_process`. The file is expected to be located in the `data/4_feature/trees`
        directory with the same name as the STEP file, but with a `.bin` extension.

        Returns
        -------
        bool
            True if the tree file exists, False otherwise.
        """
        relative_path = self._get_relative_path()
        tree_path = (cons.PATHS.DATA_FEATURE / "trees" / relative_path).with_suffix(".bin")
        return tree_path.exists()

    def _save_invariants(self):
        relative_path = self._get_relative_path()
        invariants_path = (cons.PATHS.DATA_FEATURE / "invariants" / relative_path).with_suffix(".json")
        self._invariants.to_json(invariants_path)

    def _invariants_available(self) -> bool:
        """
        Check if the invariants representation is available for the current part.

        This method checks if a JSON file containing invariants exists for the current part,
        specified by `_file_to_process`. The file is expected to be located in the `data/4_feature/invariants`
        directory with the same name as the STEP file, but with a `.json` extension.

        Returns
        -------
        bool
            True if the invariants file exists, False otherwise.
        """
        relative_path = self._get_relative_path()
        invariants_path = (cons.PATHS.DATA_FEATURE / "invariants" / relative_path).with_suffix(".json")
        return invariants_path.exists()

    def _save_vecset(self):
        """
        Save the vector set representation of the current STEP file.

        This method saves the vector set representation of the current STEP file to a file.
        The file is stored in the `data/4_feature/vecsets` directory with the same name as the STEP file,
        but with a `.npy` extension.
        """
        logging.debug("Saving vector set representation")
        relative_path = self._get_relative_path()
        vecset_path = (cons.PATHS.DATA_FEATURE / "vecsets" / relative_path).with_suffix(".npy")
        # ensure the directory exists
        vecset_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(vecset_path, self._vecset)

    def _vecset_available(self) -> bool:
        """
        Check if the vector set representation is available for the current part.

        This method checks if a vector set representation file exists for the current part,
        specified by `_file_to_process`. The file is expected to be located in the `data/4_feature/vecsets`
        directory with the same name as the STEP file, but with a `.npy` extension.

        Returns
        -------
        bool
            True if the vector set representation file exists, False otherwise.
        """
        relative_path = self._get_relative_path()
        vecset_path = (cons.PATHS.DATA_FEATURE / "vecsets" / relative_path).with_suffix(".npy")
        return vecset_path.exists()



    def _save_targets(self):
        """
        Save the extracted regression and classification features.
        """
        logger.info("Saving targets")
        if self._files_already_processed is not None:
            targets_already_processed = pd.read_csv(cons.PATHS.DATA_FEATURE / "fabwave_targets.csv")
        else:
            logger.debug("No already processed targets found, creating new DataFrame")
            logger.debug("No already processed targets found, creating new DataFrame")
            targets_already_processed = None
        new_targets = pd.DataFrame(self._targets)
        targets_all = pd.concat([targets_already_processed, new_targets], ignore_index=True)
        targets_all.to_csv(
            cons.PATHS.DATA_FEATURE / "fabwave_targets.csv", index=False
        )

    def _images_available(self) -> bool:
        """
        Check if a folder exists for the part's images.

        This method checks if a folder for the current part, specified by `_file_to_process`,
        exists in the `data/4_feature/images/fabwave` directory. The folder is expected to have
        the same name as the file to process (without the file extension).

        Returns
        -------
        bool
            True if the folder exists, False otherwise.
        """
        # TODO add folder name for part class to folder_path
        logger.info("Checking if images are available for the current part")
        # TODO add folder name for part class to folder_path
        logger.info("Checking if images are available for the current part")
        folder_name = self._file_to_process.stem
        part_class_name = self._file_to_process.parent.name
        folder_path = cons.PATHS.DATA_FEATURE / "images" / "fabwave" / part_class_name / folder_name
        return folder_path.is_dir()

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

                # convert step file to tree or invariants if possible
                tree_saved = self._tree_available()
                if not tree_saved:
                    try:
                        self._convert_to_tree()
                        self._save_tree()
                        tree_saved = True
                    except Exception as e:
                        logger.warning(f"Error converting {self._file_to_process} to TREE: {e}")
                else:
                    logger.debug(f"Tree for {self._file_to_process} already available, skipping tree-conversion")
                
                invariants_saved = self._invariants_available()
                if not invariants_saved:
                    try:
                        self._convert_to_invariants()
                        self._save_invariants()
                        invariants_saved = True
                    except Exception as e:
                        logger.warning(f"Error in converting {self._file_to_process} to INVARIANTS: {e}")
                else:
                    logger.debug(f"Invariants for {self._file_to_process} already available, skipping invariants-conversion")

                vecset_saved = self._vecset_available()
                if not vecset_saved:
                    try:
                        self._convert_to_vecset()
                        self._save_vecset()
                        vecset_saved = True
                    except Exception as e:
                        logger.warning(f"Error in converting {self._file_to_process} to VECSET: {e}")
                else:
                    logger.debug(f"Vecset for {self._file_to_process} already available, skipping vecset-conversion")
                
                # only extract targets if tree, invariants and images are
                # available
                logger.debug(f"Tree saved: {tree_saved}, Invariants saved: {invariants_saved}, Images available: {self._images_available()}, vecset available: {vecset_saved}")
                if tree_saved and invariants_saved and vecset_saved and self._images_available():
                    logger.debug("all representations available")
                    try:
                        self._get_targets()
                    except Exception as e:
                        logger.warning(f"Error extracting targets from {self._file_to_process}: {e}")

                progress_bar.update(1)

                continue

        finally:
            self._save_targets()

if __name__ == "__main__":
    pipeline = PrimaryFeaturePipeline()
    pipeline.run()
