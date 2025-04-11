from fastapi import FastAPI
from pydantic import BaseModel
import base64

from typing import Dict
import freecad
import FreeCAD
import Import 
import Part
import os, json

from pathlib import Path

app = FastAPI()

class StepFileData(BaseModel):
    filename: str
    filedata: str  # Base64-codierter String

def analyseStep(file_path: str) -> Dict[str, float]:
    
    step_extractor = StepTargetExtractor(Path(file_path))
    targets = step_extractor.get_data()

    return targets

@app.post("/upload_step/")
async def upload_step(data: StepFileData):
    file_path = f"/tmp/{data.filename}.step"

    # tmp file dor the step file
    with open(file_path, "wb") as f:
        f.write(base64.b64decode(data.filedata))

    # analyse  step file
    results = analyseStep(file_path)

    # remove the tmp file to save space
    os.remove(file_path)

    return results



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



