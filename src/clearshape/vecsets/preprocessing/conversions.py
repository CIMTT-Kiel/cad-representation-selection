import os
import shutil
import logging
from pathlib import Path
from typing import Union

import numpy as np
import open3d as o3d
import cascadio
import trimesh
import torch
import mcubes

from clearshape.vecsets.preprocessing.vecset_model import autoencoder

# Logging Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Modell laden
_MODEL_CHPT = "/clear-shape/src/clearshape/vecsets/preprocessing/vecset_model/ckpts/checkpoint-110.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vecset_encoder = autoencoder.__dict__['point_vec1024x32_dim1024_depth24_nb']()
vecset_encoder.eval()
vecset_encoder.load_state_dict(torch.load(_MODEL_CHPT, map_location='cpu', weights_only=False)['model'], strict=False)
vecset_encoder.to(DEVICE)


class CAD_Converter:
    """
    Konvertiert STEP-Dateien in STL, PLY und VecSet-Repräsentationen.
    """

    def __init__(self, step_file: Path, skip_existing: bool = False):
        self._step_file = Path(step_file)
        self._file_name = self._step_file.stem
        self._skip_existing = skip_existing

        self._tmp_dir = self._step_file.parent / "tmp"

        self._vecset_encoder = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def from_step(cls, step_file: Union[str, Path], skip_existing: bool = False) -> "CAD_Converter":
        return cls(Path(step_file), skip_existing=skip_existing)

    def to_stl(self, output_path: Union[str, Path]):
        """
        Konvertiert STEP-Datei zu STL.

        Args:
            output_path (Union[str, Path]): Vollständiger Pfad zur Zieldatei mit `.stl`-Endung.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists() and self._skip_existing:
            logger.info(f"STL-Datei existiert bereits und wird übersprungen: {output_path}")
            return

        logger.info(f"Konvertiere {self._step_file} zu STL...")
        self._tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_obj = self._tmp_dir / f"tmp.obj"
        cascadio.step_to_obj(str(self._step_file), str(tmp_obj))

        mesh = trimesh.load(tmp_obj, file_type='obj')
        mesh.export(str(output_path), file_type='stl')

        logger.info(f"STL gespeichert unter {output_path}")

    def to_ply(self, output_path: Union[str, Path]):
        """
        Konvertiert STEP-Datei zu PLY-Datei über STL-Zwischenschritt.

        Args:
            output_path (Union[str, Path]): Vollständiger Pfad zur Zieldatei mit `.ply`-Endung.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists() and self._skip_existing:
            logger.info(f"PLY-Datei existiert bereits und wird übersprungen: {output_path}")
            return

        # temporärer STL-Pfad
        self._tmp_dir.mkdir(parents=True, exist_ok=True)
        temp_stl = self._tmp_dir / f"tmp.stl"

        if not temp_stl.exists():
            self.to_stl(temp_stl)

        logger.info(f"Konvertiere STL zu PLY für {self._file_name}...")
        mesh = o3d.io.read_triangle_mesh(str(temp_stl))
        pcd = o3d.geometry.TriangleMesh.sample_points_uniformly(mesh, number_of_points=8192)
        o3d.io.write_point_cloud(str(output_path), pcd)
        logger.info(f"PLY gespeichert unter {output_path}")


    def _setup_vecset_encoder(self):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._density = 256
        self._gap = 2. / self._density

        x = np.linspace(-1, 1, self._density + 1)
        y = np.linspace(-1, 1, self._density + 1)
        z = np.linspace(-1, 1, self._density + 1)
        xv, yv, zv = np.meshgrid(x, y, z)
        self._grid = torch.from_numpy(np.stack([xv, yv, zv]).astype(np.float32)).view(3, -1).transpose(0, 1)[None].cpu()

    def to_vecset(self, output_path: Union[str, Path], export_reconstruction: bool = False):
        """
        Konvertiert STEP-Datei zu VecSet-Repräsentation (.npy) über PLY-Zwischenschritt.

        Args:
            output_path (Union[str, Path]): Vollständiger Pfad zur Zieldatei mit `.npy`-Endung.
            export_reconstruction (bool): Optionaler Export eines rekonstruierten Meshes (.stl).
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists() and self._skip_existing:
            logger.info(f"VecSet-Datei existiert bereits und wird übersprungen: {output_path}")
            return

        self._tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_ply = self._tmp_dir / f"tmp.ply"
        self.to_ply(tmp_ply)

        logger.info(f"Konvertiere PLY zu VecSet für {self._file_name}...")
        surface = trimesh.load(tmp_ply.as_posix()).vertices
        assert surface.shape[0] == 8192, "Anzahl Punkte ist ungleich 8192"

        shifts = (surface.max(axis=0) + surface.min(axis=0)) / 2
        surface = surface - shifts
        distances = np.linalg.norm(surface, axis=1)
        scale = 1 / np.max(distances)
        surface *= scale
        surface = torch.from_numpy(surface.astype(np.float32)).to(self._device)

        if self._vecset_encoder is None:
            self._setup_vecset_encoder()

        with torch.no_grad():
            if export_reconstruction:
                outputs = vecset_encoder(surface[None], self._grid.to(self._device))
                volume = outputs['o'][0].view(self._density + 1, self._density + 1, self._density + 1).permute(1, 0, 2).cpu().numpy() * (-1)
                verts, faces = mcubes.marching_cubes(volume, 0)
                verts *= self._gap
                verts -= 1.
                m = trimesh.Trimesh(verts, faces)
                m.export(output_path.with_name(output_path.stem + "_reconstruction.stl"), file_type='stl')
            else:
                outputs = vecset_encoder.encode_to_vecset(surface[None])

            np.save(output_path.as_posix(), outputs['x'].squeeze(0).cpu().numpy())
            logger.info(f"VecSet gespeichert unter {output_path}")


        shutil.rmtree(self._tmp_dir, ignore_errors=True)


