

# standard libaries
from pathlib import Path

# Third party libaries
import gmsh

# application imports
#import clearshape.step_tree.step_analysis as sta

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
                print(f"Error processing {self.path_to_step}: {e}")
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
        gmsh.initialize(interruptible=False)
        gmsh.option.setNumber("General.Terminal", 1)


        try:
            gmsh.model.add("3DMesh")
            gmsh.model.occ.importShapes(str(self.path_to_step))
            gmsh.model.occ.synchronize()

            gmsh.option.setNumber("Geometry.OCCSewFaces", 1)

            gmsh.model.occ.synchronize()

            gmsh.model.mesh.generate(3)  # 3D Vernetzung
            
            # Überprüfung, ob ein 3D-Netz generiert wurde
            
            elem_types, _, _ = gmsh.model.mesh.getElements(dim=3)

            if not elem_types:
                raise ValueError("Fehler: Kein 3D-Netz erzeugt! Möglicherweise wurde nur ein 2D-Netz erstellt.")
            
            #create dir if not exists
            self.path_to_msh.parent.mkdir(parents=True, exist_ok=True)
            gmsh.write(str(self.path_to_msh))
        except Exception as e:
            print(f"Error by meshing {self.path_to_step}: {e}")
        finally:
            gmsh.finalize()

    def _bend_path(self, orig_path, element_to_replace, new_element):
        """
        Replaces a specified element in the given path with a new element.

        Args:
            orig_path (Path): The original path object.
            element_to_replace (str): The element in the path to be replaced.
            new_element (str): The new element to replace the old element.

        Returns:
            Path: A new path object with the specified element replaced.
        """
        parts = list(orig_path.parts)
        parts[parts.index(element_to_replace)] = new_element
        return Path(*parts)


    def _build_target_path_for_msh(self, step_path):
        """
        Builds the target file path for a mesh file (.msh) based on the given step path.

        Args:
            step_path (Path): The initial path of the step file.

        Returns:
            Path: The modified path with the '3_primary' and '3_1_meshes' directories and a '.msh' file extension.
        """
        return self._bend_path(step_path, '3_primary', '3_1_meshes').with_suffix('.msh')