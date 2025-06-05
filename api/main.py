from fastapi import FastAPI
from pydantic import BaseModel
import base64

from typing import Dict
import freecad
import FreeCAD
import Import 
import Part
import os, json
import gmsh


import random
import numpy as np
import open3d as o3d

import cascadio
import trimesh

from pathlib import Path

app = FastAPI()

class StepFileData(BaseModel):
    filename: str
    filedata: str  # Base64-codierter String

def analyseStep(file_path: str) -> Dict[str, float]:
    
    step_extractor = StepTargetExtractor(Path(file_path))
    targets = step_extractor.get_data()

    return targets

@app.post("/mesh_step_file/")
async def mesh_step(data: StepFileData):
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
    step_file_path = f"/tmp/{data.filename}.step"
    msh_file_path = f"/tmp/{data.filename}.msh"

    msh_base64_data = None

    # tmp file dor the step file
    with open(step_file_path, "wb") as f:
        f.write(base64.b64decode(data.filedata))


    gmsh.initialize(interruptible=False)
    gmsh.option.setNumber("General.Terminal", 1)


    try:
        gmsh.model.add("3DMesh")
        gmsh.model.occ.importShapes(step_file_path)
        gmsh.model.occ.synchronize()

        gmsh.option.setNumber("Geometry.OCCSewFaces", 1)

        gmsh.model.occ.synchronize()

        gmsh.model.mesh.generate(3)  # 3D Vernetzung
        
        # Überprüfung, ob ein 3D-Netz generiert wurde
        
        elem_types, _, _ = gmsh.model.mesh.getElements(dim=3)

        if not elem_types:
            raise ValueError("Fehler: Kein 3D-Netz erzeugt! Möglicherweise wurde nur ein 2D-Netz erstellt.")
        
        #create dir if not exists
        Path(msh_file_path).parent.mkdir(parents=True, exist_ok=True)
        gmsh.write(msh_file_path)

        with open(msh_file_path, "rb") as file:
            msh_base64_data = base64.b64encode(file.read()).decode("utf-8")

    except Exception as e:
        print(f"Error by meshing {msh_file_path}: {e}")
    finally:
        gmsh.finalize()


    
    # remove the tmp file to save space
    if os.path.exists(msh_file_path):
        os.remove(step_file_path)
    if os.path.exists(msh_file_path):
        os.remove(msh_file_path)

    return {"msh_filedata" : msh_base64_data}

@app.post("/get_targets_from_step/")
async def get_targets(data: StepFileData):
    file_path = f"/tmp/{data.filename}.step"

    # tmp file dor the step file
    with open(file_path, "wb") as f:
        f.write(base64.b64decode(data.filedata))

    # analyse  step file
    results = analyseStep(file_path)

    # remove the tmp file to save space
    os.remove(file_path)

    return results

@app.post("/fuse_overlaps/")
def fuse_overlaps(data: StepFileData):

    fused = None
    error = False
    error_msg = None
    base64_fused_filedata = None

    # tmp file dor the step file
    step_file = decode_step_file(data)


    doc = FreeCAD.newDocument(f"{step_file}")
    try:
        Part.insert(step_file.as_posix(), doc.Name)
        shapes = [volume.Shape for volume in doc.Objects]

        assert len(shapes) > 0, f"No objects found in {step_file}"

        
        fused_shape = shapes[0] #in case shapes contain just one body fused_shape is the original shape and just will be copied

        # check if part contains multiple bodies
        if len(shapes)>1:
            for shape in shapes[1:]:
                fused_shape = fused_shape.fuse(shape)

        assert len(fused_shape.Solids) > 0, f"No solids found in {step_file}"
        
        #check if fuse process was successfull - otherwise the part contains at least two bodies which can not be fused
        if len(fused_shape.Solids)==1:
            fused = True
        else:
            fused = False 

        # Objekt im Dokument erzeugen und Shape zuweisen
        fused_obj = doc.addObject("Part::Feature", "Fused")
        fused_obj.Shape = fused_shape
        doc.recompute()

        assert len(fused_obj.Shape.Solids) > 0, f"No solids found in {step_file}"
                
        if fused:
            #save the fused shape to a file
            fused_file_path = f"/tmp/fused_{data.filename}.step"
            Part.export([fused_obj], fused_file_path)

            with open(fused_file_path, "rb") as f:
                base64_fused_filedata = base64.b64encode(f.read()).decode("utf-8")
            os.remove(fused_file_path)
    except Exception as e:
        error = True
        error_msg = e



    #build the response



    FreeCAD.closeDocument(doc.Name)
    return {"fused": fused, "filedata": base64_fused_filedata, "error": error, "error_msg" : error_msg}

@app.post("/extract_biggest_volume_as_step/")
def extract_biggest_volume_as_step(data: StepFileData):

    base64_biggest_shape_filedata = None
    error = False
    error_msg = None

    # tmp file dor the step file
    step_file = decode_step_file(data)


    doc = FreeCAD.newDocument("subpart")
    try:
        Part.insert(step_file.as_posix(), doc.Name)
        shapes = [volume.Shape for volume in doc.Objects]

        if len(shapes)>1:
            shapes.sort( key = lambda elem: elem.Volume)
            biggest_shape = shapes[-1]

            # Objekt im Dokument erzeugen und Shape zuweisen
            fused_obj = doc.addObject("Part::Feature", "Fused")
            fused_obj.Shape = biggest_shape
            doc.recompute()
        
            #save the biggest shape to a file
            biggest_shape_file_path = f"/tmp/biggest_{data.filename}.step"
            Part.export([fused_obj], biggest_shape_file_path)
            with open(biggest_shape_file_path, "rb") as file:
                base64_biggest_shape_filedata = base64.b64encode(file.read()).decode("utf-8")
            os.remove(biggest_shape_file_path)

    except Exception as e:
        error = True
        error_msg = e

    FreeCAD.closeDocument(doc.Name)

    return {"filedata": base64_biggest_shape_filedata, "error": error, "error_msg": error_msg}

@app.post("/split_miter_gear_set/")
def split_miter_gear_set(data: StepFileData):

    print("split_miter_gear_set called")
    base64_gear_shape_filedata = None
    base64_headless_screw_shape_filedata = None
    error = False
    error_msg = None

    # tmp file dor the step file
    step_file = decode_step_file(data)


    doc = FreeCAD.newDocument("subpart")
    try:
        Part.insert(step_file.as_posix(), doc.Name)
        shapes = [volume.Shape for volume in doc.Objects]
        shapes.sort( key = lambda elem: elem.Volume)

        gear = shapes[-1]
        headless_screw = shapes[0]

        # Objekt im Dokument erzeugen und Shape zuweisen
        fused_obj_gear = doc.addObject("Part::Feature", "Fused")
        fused_obj_gear.Shape = gear
        doc.recompute()

        fused_obj_hs = doc.addObject("Part::Feature", "Fused")
        fused_obj_hs.Shape = headless_screw
        doc.recompute()


        gear_file = Path(step_file.parent / f"{step_file.stem}_EXTRACTED_Gears.step")
        headless_screw_file = Path(step_file.parent / f"{step_file.stem}_EXTRACTED_HeadlessScrews.step")

        #save the biggest shape to a file
        Part.export([fused_obj_gear], gear_file.as_posix())
        Part.export([fused_obj_hs], headless_screw_file.as_posix())
        
        #encode step_files
        with open(gear_file.as_posix(), "rb") as file:
            base64_gear_shape_filedata = base64.b64encode(file.read()).decode("utf-8")
        with open(headless_screw_file.as_posix(), "rb") as file:
            base64_headless_screw_shape_filedata = base64.b64encode(file.read()).decode("utf-8")
        os.remove(gear_file.as_posix())
        os.remove(headless_screw_file.as_posix())

    except Exception as e:
        error = True
        error_msg = e

    return {"filedata_gear": base64_gear_shape_filedata, "filedata_headless_screw" : base64_headless_screw_shape_filedata, "error": error, "error_msg": error_msg}

def decode_step_file(data: StepFileData) -> str:
    """
    Decodes a base64-encoded STEP file and saves it to a temporary location.
    """
    step_file_path = Path(f"/tmp/{data.filename}.step")

    with open(step_file_path.as_posix(), "wb") as f:
        f.write(base64.b64decode(data.filedata))

    return step_file_path

@app.post("/step_to_ply/")
async def step_to_ply(data: StepFileData):
    """

    """
    error = False
    error_msg = None
    ply_base64_data = None

    step_file_path = Path(f"/tmp/{data.filename}.step")
    ply_file_path = Path(f"/tmp/{data.filename}.ply")

    output_dir = Path(f"/tmp")
    #create dir if not exists
    step_file_path.parent.mkdir(parents=True, exist_ok=True)

    # tmp file dor the step file
    with open(step_file_path, "wb") as f:
        f.write(base64.b64decode(data.filedata))

    try:
        print(f"step_file_path: {step_file_path}")
        step_converter = STEP_Converter(step_file_path, Path(f"/tmp/"))
        step_converter.to_ply()
        #create dir if not exists
        print("try to write out ply file..")
        with open(ply_file_path, "rb") as file:
            ply_base64_data = base64.b64encode(file.read()).decode("utf-8")

    except Exception as e:
        error = True
        error_msg = str(e)
        print(f"Error by converting step to ply in file {data.filename}: {e}")

    
    
    # remove the tmp file to save space
    os.remove(step_file_path)
    os.remove(step_converter.stl_file.as_posix())
    os.remove(step_converter.ply_file.as_posix())

    return {"ply_filedata" : ply_base64_data}

class STEP_Converter():

    def __init__(self, step_file, output_dir):
        self._step_file=Path(step_file)
        self.file_name = step_file.stem
        self._output_dir = Path(output_dir)
        self.pcd = None


    # def to_stl_(self):
    #     """
    #     Generates a 3D mesh from a STEP file using Gmsh and saves it to a specified path.
    #     This method initializes Gmsh, imports a STEP file, generates a 3D mesh, and saves the mesh to a specified file path.
    #     It ensures that a 3D mesh is created and handles any exceptions that occur during the process.
    #     Raises:
    #         ValueError: If no 3D mesh is generated, indicating that only a 2D mesh may have been created.
    #         Exception: For any other errors that occur during the meshing process.
    #     Side Effects:
    #         Creates directories if they do not exist.
    #         Writes the generated mesh to a file.
    #     """
    #     gmsh.initialize(interruptible=False)
    #     gmsh.option.setNumber("General.Terminal", 1)


    #     try:
    #         gmsh.model.add("3DMesh")
    #         gmsh.model.occ.importShapes(str(self._step_file))
    #         gmsh.model.occ.synchronize()

    #         gmsh.option.setNumber("Geometry.OCCSewFaces", 1)

    #         gmsh.model.occ.synchronize()

    #         gmsh.model.mesh.generate(3)  # 3D Vernetzung
            
    #         # Überprüfung, ob ein 3D-Netz generiert wurde
            
    #         elem_types, _, _ = gmsh.model.mesh.getElements(dim=3)

    #         if not elem_types:
    #             raise ValueError("Fehler: Kein 3D-Netz erzeugt! Möglicherweise wurde nur ein 2D-Netz erstellt.")
            
    #         #create dir if not exists
    #         gmsh.write(str(self._output_dir / f"mesh/{self.file_name}.stl"))
    #     except Exception as e:
    #         print(f"Error by meshing {self.path_to_step}: {e}")
    #     finally:
    #         gmsh.finalize()

    def to_stl(self):
        print(f"to stl: {self._step_file}")
        tmp_path = Path("tmp")
        tmp_path.mkdir(parents=True, exist_ok=True)
        tmp_file = tmp_path / "out.obj"
        print("Convert STEP to OBJ..")
        cascadio.step_to_obj(str(self._step_file), tmp_file.as_posix())

        # Load the OBJ file using trimesh
        print("Convert OBJ to STL..")
        print(f"tmp_file: {tmp_file}")
        mesh = trimesh.load(tmp_file.as_posix())
        print("Export to STL..")
        mesh.export(str(self._output_dir / f"{self.file_name}.stl"), file_type='stl')

        # Remove the temporary OBJ file
        if tmp_file.exists():
            os.remove(tmp_file)


    def to_ply(self):
        self.stl_file = self._output_dir / f"{self.file_name}.stl"
        self.ply_file = self._output_dir / f"{self.file_name}.ply"

        print(f"stl_file: {self.stl_file}")
        print(f"ply_file: {self.ply_file}")

        #create dir if not exists
        self.stl_file.parent.mkdir(parents=True, exist_ok=True)
        self.ply_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.to_stl()
        
        #load stl file
        print(f"load stl file: {self.stl_file}")
        mesh = o3d.io.read_triangle_mesh(str(self.stl_file))
        print(f"create point cloud from mesh: {mesh}")
        pcd = o3d.geometry.TriangleMesh.sample_points_uniformly(mesh, number_of_points=8192)

        #save to ply
        print(f"save point cloud to ply file: {self.ply_file}")
        o3d.io.write_point_cloud(str(self.ply_file), pcd)

        self.pcd = pcd

class StepTargetExtractor:
    def __init__(self, path:Path):


        doc = FreeCAD.newDocument()
        Part.insert(path.as_posix(), doc.Name)

        self.total_faces = 0
        self.total_edges = 0
        self.total_vertices = 0
        self.volume = 0
        self.multiple_shape_indicator = 0

        #initialize list of bounding box values for each shape
        self.xmins = []
        self.ymins = []
        self.zmins = []

        self.xmaxs = []
        self.ymaxs = []
        self.zmaxs = []

        

        for obj in doc.Objects:
            if hasattr(obj, "Shape"):
                self.total_faces += len(obj.Shape.Faces)
                self.total_edges += len(obj.Shape.Edges)
                self.total_vertices += len(obj.Shape.Vertexes)
                self.volume += obj.Shape.Volume
                self.multiple_shape_indicator += 1

                self.bounds = obj.Shape.BoundBox

                self.xmins.append(self.bounds.XMin)
                self.ymins.append(self.bounds.YMin)
                self.zmins.append(self.bounds.ZMin)
                
                self.xmaxs.append(self.bounds.XMax)
                self.ymaxs.append(self.bounds.YMax)
                self.zmaxs.append(self.bounds.ZMax)

            else:
                raise ValueError(f"No shapes found in {path}")
        
        
        #claulate the bounding box of the whole file
        self.xmin = min(self.xmins)
        self.ymin = min(self.ymins)
        self.zmin = min(self.zmins)

        self.xmax = max(self.xmaxs)
        self.ymax = max(self.ymaxs)
        self.zmax = max(self.zmaxs)

        self.dx = self.xmax - self.xmin
        self.dy = self.ymax - self.ymin
        self.dz = self.zmax - self.zmin

    def get_step_units(file_path):
        with open(file_path, 'r') as step_file:
            for line in step_file:
                # Suche nach Zeilen, die die Einheit definieren
                if "SI_UNIT" in line:
                    # Prüfe, ob die Einheit Millimeter ist
                    if ".MILLI." in line and ".METRE." in line:
                        return "mm"
                    # Prüfe, ob die Einheit Meter ist
                    elif ".METRE." in line:
                        return "m"
                    # Prüfe, ob die Einheit Zoll ist
                    elif ".INCH." in line:
                        return "inches"
                    # Prüfe, ob die Einheit Fuß ist
                    elif ".FOOT." in line:
                        return "feet"
        # Wenn keine Einheit gefunden wurde, Standardwert auf Millimeter setzen
        return "mm"
    
    def get_data(self):
        return {"faces" : self.total_faces, "edges" : self.total_edges, "vertices" : self.total_vertices, "volume" : self.volume, "dx" : self.dx, "dy" : self.dy, "dz" : self.dz}



