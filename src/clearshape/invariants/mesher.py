

# standard libaries
from pathlib import Path

# Third party libaries
import gmsh

# application imports
#import clearshape.step_tree.step_analysis as sta

class StepMesher:

    def __init__(self, path : str):
        self.path_to_step = path
    
    @classmethod
    def mesh_from_step(cls, path : str):
        """Creates a 'StepMesher' instance from a path to a step file.
         
          The method reads the step file specified in path and returns the 3D tetraeder-mesh as a numpy array. The method checks if the file was already meshed. If not the geometry is tried to be meshed via gmsh module."""
        
        cls = cls(path)
        cls._process_file(path)

        return cls 
        
    
    def _process_file(self, path : str):
        self.path_to_msh = self._build_target_path_for_msh(path)
        if not self.path_to_msh.exists() or self.path_to_msh.stat().st_size == 0:
            self.path_to_msh.parent.mkdir(parents=True, exist_ok=True)
            try:
                self._mesh_step_file()
            except Exception as e:
                print(f"Error processing {self.path_to_step}: {e}")
        return None
    
    def _mesh_step_file(self):
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
        parts = list(orig_path.parts)
        parts[parts.index(element_to_replace)] = new_element
        return Path(*parts)


    def _build_target_path_for_msh(self, step_path):
        return self._bend_path(step_path, '3_primary', '3_1_meshes').with_suffix('.msh')