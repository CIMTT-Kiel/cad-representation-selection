import subprocess
import sys

from clearshape.constants import PATHS

# Liste der Python-Skripte, die nacheinander ausgeführt werden sollen
scripts = [
    #PATHS.ROOT / "src/clearshape/pipelines/model_input_to_models_invs_cls.py",
    PATHS.ROOT / "src/clearshape/pipelines/model_input_to_models_vecsets_cls.py",
    PATHS.ROOT / "src/clearshape/pipelines/model_input_to_models_invs_reg.py",
    PATHS.ROOT / "src/clearshape/pipelines/model_input_to_models_vecsets_reg.py", 
]

def run_scripts():
    for script in scripts:
        print(f"\nStarte {script} ...")
        try:
            # subprocess.run führt das Script im aktuellen Interpreter aus
            subprocess.run([sys.executable, script], check=True)
            print(f"✅ {script} erfolgreich beendet.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Fehler beim Ausführen von {script}: {e}")
            break

if __name__ == "__main__":
    run_scripts()
