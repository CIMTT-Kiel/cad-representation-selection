from PIL import Image
from pathlib import Path
ROOT = Path("/Users/lassepaplow/Source/clear-shape/data/4_feature")
PICTURES_DIR = ROOT/"images_0"
image_paths = sorted(list(PICTURES_DIR.rglob("*.png"))) 


for image in image_paths:
    
  
    # Opening the primary image (used in background) 
    
    img1 = Image.open(image) 

    new_parts = []
    for part in image.parts:
        if part == 'images_0':
            part = 'images_1'
        new_parts.append(part)

    # Neuen Pfad erstellen
    image2 = Path(*new_parts)
     
  
    # Opening the secondary image (overlay image) 
    img2 = Image.open(image2) 
    
  
    # Pasting img2 image on top of img1  
    # starting at coordinates (0, 0) 
    img1.paste(img2, (0,0), mask = img2) 

    new_parts = []
    for part in image.parts:
        if part == 'images_0':
            part = 'images'
        new_parts.append(part)

    # Neuen Pfad erstellen
    image_path = Path(*new_parts)
    
    image_path.parent.mkdir(exist_ok=True, parents=True)
    # Saving the image 
    img1.save(image_path)