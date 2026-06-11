
import requests
import base64
from pathlib import Path
import logging


# set up logger
logging_level = logging.DEBUG
logger = logging.getLogger(__name__)
logger.setLevel(logging_level)

class RegressionTargetExtractor:

    API_URL = "http://step_api:8000/get_targets_from_step/"


    def __init__(self, path : Path):
        self.path_to_step = path
        self.target_dict = None

    @classmethod
    def analyze_step(cls, path : Path):

        cls = cls(path)
        cls._process_file(cls.path_to_step)

        return cls, cls.target_dict
    
    def _process_file(self, path_to_step: Path) -> dict:

        with open(path_to_step.as_posix(), "rb") as file:
            base64_data = base64.b64encode(file.read()).decode("utf-8")

        payload = {
            "filename": path_to_step.stem,
            "filedata": base64_data
        }

        response = requests.post(self.API_URL, json=payload)

        if response.status_code == 200:
            self.target_dict =  response.json()
        else:
            logger.warning("Fehler:", response.text)

    
if __name__ == "__main__":
    # Example usage
    path_to_step = Path("/workspaces/data/3_primary/fabwave/Idler Sprocket/8b9b7f17-2ed6-452b-aa2d-9d72e13ca3e7 copy 2.step") # Hier zum testen eine lokale STEP-Datei angeben
    extractor, targets = RegressionTargetExtractor.analyze_step(path_to_step)
    print(targets)


    