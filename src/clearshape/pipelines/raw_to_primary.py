import os
import logging
from typing import List, Dict, Tuple
from collections import Counter

import requests
import base64
# import freecad
# import FreeCAD
# import Import
# import Part
import os, shutil, json, time
from pathlib import Path
import pandas as pd

from clearshape import constants

# Logger konfigurieren
logging.basicConfig(level=logging.INFO)

class RawPrimaryPipeline:

    def __init__(self, config: Dict):
        self.approx_timestamp = time.strftime("%Y-%m-%d-%H:%M")
        self.config = config

        #for monitoring only
        self.skiped_files_counter = 0 
        self.files_splited = 0
        self.error_files = {}
        self.files_excluded_due_to_total_class_size = 0
        self.skip_existing = config["skip_existing"]

    def load_data(self, path_to_raw_data: str) -> List:
        """Load raw step files"""
        logging.info(f"Load STEP raw files from {path_to_raw_data}..")
        raw_step_files=list(path_to_raw_data.rglob('*.step'))+list(path_to_raw_data.rglob('*.STEP'))+list(path_to_raw_data.rglob('*.stp'))
        raw_step_files = raw_step_files 
        logging.info(f"load {len(raw_step_files)} files.")
        return raw_step_files
    
    # def build_target_file(self, step_file: Path):
    #     target_filename  = Path(f"{step_file.stem}.step") #only to check if file was already processed
    #     target_dir = Path(str(step_file).replace("1_raw", "3_primary")).parent

    #     target_file = target_dir / target_filename 

    #     return target_file


    def fuse_overlapping_bodies(self, raw_step_files: List) -> Tuple[Dict]:
        """Try to fuse bodies, if step file contains multiple closedShell Objects. If the parts are not connected the fused part will again contain multiple solids. The function returns a tuple of successfully fused parts and remaining multibody files"""
        logging.debug(" Fuse overlapping bodies")

        fused_step_files = []
        multibody_step_files = []
        for i, step_file in enumerate(raw_step_files):
            logging.info(f" ->[{i+1}|{len(raw_step_files)}]")

            target_file = self.generate_target_path(step_file)

            if target_file.exists() and self.skip_existing:
                logging.debug(f" -> skip file as it already exists")
                pass

            else:
                with open(step_file.as_posix(), "rb") as file:
                    base64_data = base64.b64encode(file.read()).decode("utf-8")

                payload = {
                    "filename": target_file.stem,
                    "filedata": base64_data
                }

                response = requests.post("http://step_api:8000/fuse_overlaps/", json=payload)

                if response.status_code == 200:
                    response_dict =  response.json()
                else:
                    logging.warning("Error while posting to RestAPI with code:", response.status_code)


                #process the response
                if response_dict["fused"] == True:
                    #save the fused shape to a file
                    with open(target_file.as_posix(), "wb") as f:
                        f.write(base64.b64decode(response_dict["filedata"]))
                        
                    fused_step_files.append(step_file)

                elif response_dict["fused"] == False:
                    #save the original file to a new location
                    shutil.copy(step_file.as_posix(), target_file.as_posix())
                    multibody_step_files.append(step_file)

                else:
                    logging.warning("Error while posting to RestAPI!")
                    self.error_files[step_file] = "Error while posting to RestAPI"
        
        return (fused_step_files, multibody_step_files)
    
    def merge_multibodie_parts_if_possible(self,fused_files : List, multibody_parts : List):
        """Try to fix multibody parts by separating the bodies by volume. The rules which and if to keep differs by the class. These rule are hardcoded based on previous observations"""
        logging.info(" Try to merge reamining multibodie parts")

        files_pool= fused_files.copy()

        for file in multibody_parts:
            #check if the part should be handled by max_volume rule
            if True in [cl in str(file) for cl in self.config["classes_to_apply_max_volume_rule"]]:
                try:
                    with open(file.as_posix(), "rb") as file:
                        base64_data = base64.b64encode(file.read()).decode("utf-8")

                    payload = {
                        "filename": file.stem,
                        "filedata": base64_data
                    }

                    response = requests.post("http://step_api:8000/extract_biggest_volume_as_step/", json=payload)

                    if response.status_code == 200:
                        response_dict =  response.json()
                    else:
                        logging.warning("Error while posting to RestAPI:", response.text)

                    #process the response
                    if response_dict["error"] == True:
                        self.error_files[file]= response_dict["error_msg"]
                    else:
                        #save the fused shape to a file
                        target_file = self.generate_target_path(file)
                        with open(target_file.as_posix(), "wb") as f:
                            f.write(base64.b64decode(response_dict["filedata"]))

                        files_pool.append(file) 

                except Exception as e:
                    self.error_files[file] = e

            elif "Miter Gear Set Screw" in str(file) and self.config["split_Miter_Gear_Set_Screw"]:
                #special rule for Miter Gear Set - Both parts are kept and sorted to the classes headless screws and Gears. These files will be marked with {NEW_CLASS}_EXTRACTED at the end of filename and will be handled in func generate_target_files

                try:
                    with open(file.as_posix(), "rb") as file:
                        base64_data = base64.b64encode(file.read()).decode("utf-8")

                    payload = {
                        "filename": file.stem,
                        "filedata": base64_data
                    }

                    response = requests.post("http://step_api:8000/split_miter_gear_set/", json=payload)

                    if response.status_code == 200:
                        response_dict =  response.json()
                    else:
                        logging.warning("Error while posting to RestAPI:", response.text)

                    #process the response
                    if response_dict["error"] == True:
                        self.error_files[file]= response_dict["error_msg"]
                    else:
                        #save the splitted shapes to files

                        #gear shape
                        gear_target_file = self.generate_target_path(Path(file.parent / f"{file.stem}_EXTRACTED_Gears.step"))
                        with open(gear_target_file.as_posix(), "wb") as f:
                            f.write(base64.b64decode(response_dict["filedata_gear"]))

                        headless_screw_target_file = self.generate_target_path(Path(file.parent / f"{file.stem}_EXTRACTED_HeadlessScrews.step"))
                        with open(headless_screw_target_file.as_posix(), "wb") as f:
                            f.write(base64.b64decode(response_dict["filedata_headless_screw"]))

                        files_pool.append(file)

                except Exception as e:
                    self.error_files[file] = e


            else:
                self.error_files[file] = "Multibody part with no specified rule to keep one valid closed shell"
            
        return files_pool

    def generate_target_path(self, file : Path)->Path:

        """generate the target path for the primary data"""
        logging.debug(f"generate target path for {file}..")

        primary_data = {}
            
        target_file=Path(str(file).replace("1_raw", "3_primary").replace("STEP",""))

        target_dir = target_file.parent
        target_filename = f"{target_file.stem}.step"

        target_file = target_dir / target_filename
        

        #catch the Extracted parts
        if "EXTRACTED" in str(file):
            new_class = str(target_file).split("_")[-1].split(".")[0]
            dirs = list(target_file.parts)
            idx_fabwave = dirs.index("fabwave")
            dirs[idx_fabwave+1] = new_class

            target_file = Path(*dirs)

        logging.debug(f" -> {target_file}")

        #create the target directory if it does not exist
        target_dir.mkdir(exist_ok=True, parents=True)

        return target_file

    def filter_data(self, files_pool_unfiltered : List):
        logging.info("Filter data by given criterions..")

        #setup
        files_pool = files_pool_unfiltered.copy()
        report_dir = constants.PATHS.ROOT / "reports/raw->primary"
        report_dir.mkdir(exist_ok=True, parents=True)



        #1. filter classes by total samples
        min_size = self.config["filter_criteria"]["min_class_size"]
        classes_to_exclude_due_to_size = []

        processed_files_classes = [file.parent for file in files_pool] #files processed in this run

        cl_size = Counter(file.stem for file in processed_files_classes) #Size of each existing class

        #write cl sizes json to reports
        with open(constants.PATHS.ROOT / f'reports/raw->primary/class_sizes_before_filter_{self.approx_timestamp}.json', 'w') as fp:
            json.dump(cl_size, fp)

        classes_to_exclude_due_to_size = [cl for cl in cl_size.keys() if cl_size[cl]<min_size]

        if classes_to_exclude_due_to_size:
            files_pool = self.exclude_classes(files_pool, classes_to_exclude_due_to_size)

        #write class sizes after filter to json
        with open(constants.PATHS.ROOT / f'reports/raw->primary/class_sizes_after_filter_{self.approx_timestamp}.json', 'w') as fp:
            json.dump(Counter(file.parent.stem for file in files_pool), fp)

        #2. exclude classes manually if given
        if self.config["filter_criteria"]["classes_to_exclude"]:
            files_pool = self.exclude_classes(files_pool, self.config["filter_criteria"]["classes_to_exclude"] )

        #3. remove files from duplicates file
        duplicates_file = constants.PATHS.ROOT / f'reports/raw->primary/detected_duplicates.csv'
        if duplicates_file.exists():
            dplk = pd.read_csv(duplicates_file, header=None, names=['file', 'error'])
            for file in list(dplk.file):
                if Path(file).exists():
                    os.remove(file)

        #DELETE ALL FILTERED FILES
        filtered_files = set(files_pool_unfiltered)-set(files_pool)
        logging.info(f" -> delete {len(filtered_files)} due to filter criteria")
        for file in filtered_files:
            if file.exists():
                os.remove(file)

        return files_pool
    
    def exclude_classes(self, data : List, classes_to_exclude : List):
        
        #get keys to delete
        for cl in classes_to_exclude:
            logging.info(f"  exclude {cl} due to total size..")
            keys_to_delete = [key for key in data if cl in str(key)]

            for del_key in keys_to_delete:
                self.files_excluded_due_to_total_class_size+=1
                self.error_files[del_key] = f"Excluded class {cl} due to total of samples <min_size specified in config"
                del data[del_key]

        return data

    def save_data(self, data: List[Dict]):
        """save the shapes to primary as step"""
        logging.info(f"save the shapes to primary state as step")

        for path_item in data.keys():
            target_dir = path_item.parent
            target_dir.mkdir(exist_ok=True, parents=True)
            data[path_item].exportStep(path_item.as_posix())

    def run(self):
        """run the pipline"""
        raw_data = self.load_data(self.config["path_to_raw"])

        files_pool, multibody_files = self.fuse_overlapping_bodies(raw_data)
        logging.info(f" fused successfully: {len(files_pool)} remaining multibody parts: {len(multibody_files)}")
        files_pool = self.merge_multibodie_parts_if_possible(files_pool, multibody_files)
        logging.info(f" After merge datapool: {len(files_pool)} Discarded: {len(self.error_files.keys())}")
        #primary_files = self.generate_target_paths(files_pool)
        #logging.info(f" After reodering: {len(primary_files.keys())}")
        primary_files = self.filter_data(files_pool)



        #self.save_data(primary_files)
        self.export_error_file()

        logging.info(f"Report: OK: {len(primary_files)}, failed or excluded: {len(self.error_files.keys())}, handled: {self.files_splited} CHECKSUM: {len(primary_files)-self.files_splited+len(self.error_files.keys())}")
        
    def export_error_file(self):
        error_file = constants.PATHS.ROOT / f"reports/raw->primary/excluded_files_{self.approx_timestamp}.csv"
        error_file.parent.mkdir(exist_ok=True, parents=True)
        with open(error_file.as_posix(), "w") as f:
            for key in self.error_files:
                file = key
                error = self.error_files[key]
                f.write(f"{file},{error}\n")



# Konfiguration
path_to_raw_data = constants.PATHS.DATA_RAW 
path_to_primary = constants.PATHS.DATA_PRIMARY

config = {
    "path_to_raw": path_to_raw_data,
    "path_to_primary" : path_to_primary,
    "split_Miter_Gear_Set_Screw" : True,
    "classes_to_apply_max_volume_rule" : ["Boxes", "Pipes", "Pipe_Joints", "Sprocket Taper-Lock Bushing"], #None if rule should not be applied
    "filter_criteria": {
        "min_class_size": 10,
        "classes_to_exclude" : None #not implemented yet
    },
    "skip_existing": True
}

# Pipeline ausführen
constants.PATHS.DATA_RAW
pipeline = RawPrimaryPipeline(config)
pipeline.run()

