import math
import random
from pathlib import Path
import os

import freecad
import FreeCAD
import FreeCADGui
import Import
import Mesh
import numpy as np

from pivy import coin
import csv

RESOLUTION = 224
BACKGROUND = 'White'  # 'White', 'Black', 'Transparent', 'Current"

def load_model(path: str):
    """
    Load a model from a step file into a FreeCAD document.

    Args:
        path (str): The file location of the step file.

    Returns:
        freecad_document (FreeCAD.Document)
    """
    Import.open(path)
    return FreeCAD.ActiveDocument

ROOT = Path("/Users/lassepaplow/Source/clear-shape/data")
MODELS_DIR = ROOT / "3_primary/fabwave"
PICTURES_DIR = ROOT / "4_feature/images"

model_paths = sorted(list(MODELS_DIR.rglob("*.step")))


gr = (1 + math.sqrt(5)) / 2
camera_views = [
    [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1], [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
    [0, 1 / gr, gr], [0, 1 / gr, -gr], [0, -1 / gr, gr], [0, -1 / gr, -gr],
    [gr, 0, 1 / gr], [gr, 0, -1 / gr], [-gr, 0, 1 / gr], [-gr, 0, -1 / gr],
    [1 / gr, gr, 0], [-1 / gr, gr, 0], [1 / gr, -gr, 0], [-1 / gr, -gr, 0]
]

ART_STYLE = [5, 2]

zoom_dic = {}
count = 0

try:
    for art in ART_STYLE:
        image_dir = f'images_{count}'
        PICTURES_DIR = ROOT / '4_feature' / image_dir

        volumes = []
        index = 0

        for model_path in model_paths:
            image_target_path = PICTURES_DIR / model_path.relative_to(MODELS_DIR).with_suffix("")
            image_search_path = ROOT / 'images' / model_path.relative_to(MODELS_DIR).with_suffix("")

            if image_search_path.is_dir() and len(list(image_search_path.glob("*.png"))) >= 20:
                print('Already has 20 images')
                continue
            
            # if os.path.isdir(image_target_path):
            #     print('schon vorhanden')
            #     continue
            # print(image_target_path)

            try:
                load_model(model_path.as_posix())
            except:
                continue
        
            FreeCADGui.ActiveDocument.ActiveView.setCameraType('Orthographic')
            doc = FreeCAD.ActiveDocument
            view = FreeCADGui.ActiveDocument.ActiveView
            camera = FreeCADGui.ActiveDocument.ActiveView.getCameraNode()
            placement_original = doc.RootObjects[0].Placement
            FreeCADGui.runCommand('Std_DrawStyle', art)

            for image_index in range(len(camera_views)):
                target_object = doc.RootObjects[0]
                FreeCADGui.ActiveDocument.ActiveView.fitAll()
                pos = FreeCAD.Vector((camera_views[image_index][0], camera_views[image_index][1], camera_views[image_index][2]))
                camera.position.setValue(pos)
                camera.pointAt(coin.SbVec3f(0.0, 0.0, 0.0))
                FreeCADGui.ActiveDocument.ActiveView.fitAll()

                image_target_path = PICTURES_DIR / model_path.relative_to(MODELS_DIR).with_suffix("")
                new_file_name = image_target_path.name + f"_#{image_index}.png"
                image_target_path = image_target_path / new_file_name
                image_target_path.parent.mkdir(exist_ok=True, parents=True)

                error = 0

                if art == 2:
                    BACKGROUND = 'Transparent'
                try:
                    view.saveImage(
                        image_target_path.as_posix(),
                        448,
                        448,
                        BACKGROUND,
                    )

                    # Check if the first image is black
                    from PIL import Image
                
                    with Image.open(image_target_path.as_posix()) as img:
                        if not img.getbbox():
                            error += 1
                    if error == 10:
                        raise Exception("Pictures are black.")
                except:
                    raise

                index += 1

            FreeCAD.closeDocument(doc.Name)

        count += 1
        print(count)
except:
    raise