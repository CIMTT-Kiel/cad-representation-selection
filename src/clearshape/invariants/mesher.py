

# standard libaries
from pathlib import Path
import base64
import requests
import logging

#custom libraries
import clearshape.constants as constants
#from clearshape.constants import API_URL

# set up logger
logging_level = logging.ERROR
logger = logging.getLogger(__name__)
logger.setLevel(logging_level)

class StepMesher:
    """
        Represents a mesher for STEP files, which generates a 3D tetrahedral mesh using the gmsh library.

        Parameters
        ----------
        path : str
            The path to the STEP file to be meshed.

        Attributes
        ----------
        path_to_step : str
            Stores the path to the STEP file.
        path_to_msh : str
            Stores the path to the generated mesh file.

        Examples
        --------
        >>> # create StepMesher instance from STEP file and generate mesh
        >>> mesher = StepMesher.mesh_from_step("path.stp")
        >>> # access the path to the generated mesh file
        >>> mesher.path_to_msh
    """

    def __init__(self, path : str):
        self.path_to_step = path
    
    @classmethod
    def mesh_from_step(cls, path : str):
        """
        Creates a `StepMesher` instance from a STEP file located at the specified path and generates a 3D mesh.

        This method initializes a `StepMesher` instance, processes the STEP file to generate a 3D tetrahedral mesh using the gmsh library, and stores the path to the generated mesh file.

        Parameters
        ----------
        path : str
            The file system path to the STEP file that is to be meshed.

        Returns
        -------
        StepMesher
            An instance of the `StepMesher` class with the mesh generated and the path to the mesh file stored.

        Examples
        --------
        >>> mesher = StepMesher.mesh_from_step('path/to/your/step_file.stp')

        Notes
        -----
        This method assumes the STEP file is well-formed and that the gmsh library is properly installed and configured.
        """
        instance = cls(path)
        instance._process_file(path)
        return instance 
        
    
    def _process_file(self, path : str):
        """
        Processes the given file path to generate a mesh file.

        This method builds the target path for the mesh file and checks if the file
        already exists or if its size is zero. If the file does not exist or is empty,
        it creates the necessary directories and attempts to generate the mesh file.

        Args:
            path (str): The file path to process.

        Returns:
            None

        Raises:
            Exception: If an error occurs during the mesh generation process.
        """
        self.path_to_msh = self._build_target_path_for_msh(path)
        if not self.path_to_msh.exists() or self.path_to_msh.stat().st_size == 0:
            self.path_to_msh.parent.mkdir(parents=True, exist_ok=True)
            try:
                self._mesh_step_file()
            except Exception as e:
                logger.warning(f"Error processing {self.path_to_step}: {e}")
        return None
    
    def _mesh_step_file(self):
        """
        Generates a 3D mesh from a STEP file using Gmsh and saves it to a specified path.
        This method initializes Gmsh, imports a STEP file, generates a 3D mesh, and saves the mesh to a specified file path.
        It ensures that a 3D mesh is created and handles any exceptions that occur during the process.
        Raises:
            ValueError: If no 3D mesh is generated, indicating that only a 2D mesh may have been created.
            Exception: For any other errors that occur during the meshing process.
        Side Effects:
            Creates directories if they do not exist.
            Writes the generated mesh to a file.
        """

        with open(self.path_to_step.as_posix(), "rb") as file:
            base64_data = base64.b64encode(file.read()).decode("utf-8")

        payload = {
            "filename": self.path_to_step.stem,
            "filedata": base64_data
        }

        response = requests.post("http://step_api:8000/mesh_step_file/", json=payload)

        logger.debug("Response status code:", response.status_code)
        if response.status_code == 200:
            mesh_data_dict =  response.json()
            with open(str(self.path_to_msh), "wb") as f:
                logger.debug("Decode the file data and write it to the file..")
                f.write(base64.b64decode(mesh_data_dict["msh_filedata"]))
        else:
            logger.warning("Fehler:", response.text)

    def _build_target_path_for_msh(self, step_path):
        """
        Constructs the target file path for the mesh file (.msh) based on the given STEP file path.

        This method takes the path of a STEP file and transforms it into the corresponding path for the mesh file.
        It replaces the directory structure to point to the 'fabwave_meshes' directory and changes the file extension to '.msh'.

        Parameters
        ----------
        step_path : Path
            The original path of the STEP file.

        Returns
        -------
        Path
            The constructed path for the mesh file with the updated directory structure and '.msh' extension.
        """
        relative_path = step_path.relative_to(constants.PATHS.DATA_PRIMARY / "fabwave")
        target_path = (constants.PATHS.DATA_PRIMARY / "fabwave_meshes" / relative_path).with_suffix('.msh')
        return target_path
    

#check functionality
if __name__ == "__main__":
    # Example usage
    path_to_step = Path("/workspaces/data/3_primary/fabwave/Boxes/0a9450c1-da70-4432-9b3b-12d1b7af8da5.step") # Hier zum testen eine lokale STEP-Datei angeben
    mesher = StepMesher.mesh_from_step(path_to_step)
    print(mesher.path_to_msh)