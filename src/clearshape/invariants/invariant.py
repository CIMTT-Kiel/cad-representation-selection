# standard libaries
from pathlib import Path
import os
import logging

# Third party libaries
import meshio
import numpy as np
import json
import quadpy
from sklearn.decomposition import PCA 

# application imports
from clearshape.invariants.mesher import StepMesher

# set up logger
logging_level = logging.DEBUG
logger = logging.getLogger(__name__)
logger.setLevel(logging_level)

class InvariantCalculator:
    """
        A class to calculate invariants from a STEP file mesh using the gmsh library.

        Attributes
        ----------
        moment_permutations : list
            A list of moment permutations used for calculations.
        eps : float
            A small epsilon value to avoid division by zero.
        path : Path
            The path to the mesh file.
        mues : dict
            A dictionary to store calculated mues.
        pis : dict
            A dictionary to store calculated pis.

        Methods
        -------
        calculate_invariants_from_step(path : Path)
            Class method to create an InvariantCalculator instance from a path to a mesh file and calculate invariants.
        _process_file(normalized=False)
            Processes the mesh file to calculate mues and pis.
        to_json(path : Path)
            Saves the calculated invariants to a JSON file.
    """


    moment_permutations =  [[0, 0, 0],
                            [1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1],
                            [2, 0, 0],
                            [0, 2, 0],
                            [0, 0, 2],
                            [2, 1, 0],
                            [2, 0, 1],
                            [1, 2, 0],
                            [1, 0, 2],
                            [0, 2, 1],
                            [0, 1, 2],
                            [3, 0, 0],
                            [0, 3, 0],
                            [0, 0, 3],    
                            [3, 0, 3],
                            [3, 3, 0],
                            [0, 3, 3]]
    
    eps = 1e-12

    def __init__(self, path : Path):
        self.path = path
        self.mues = dict()
        self.pis = dict()

    @classmethod
    def calculate_invariants_from_step(cls, path : Path):
        """
            Creates an `InvariantCalculator` instance from a STEP file located at the specified path and calculates invariants.

            This method initializes a `StepMesher` instance, processes the STEP file to generate a 3D tetrahedral mesh using the gmsh library, and calculates the invariants from the generated mesh.

            Parameters
            ----------
            path : Path
                The file system path to the STEP file that is to be meshed.
            mues : dict
                A dictionary to store calculated mues.
            pies : dict
                A dictionary to store calculated pies.

            Returns
            -------
            InvariantCalculator
                An instance of the `InvariantCalculator` class with the invariants calculated and stored in the dictionaries mues for the moments and pies for the invariants.

            Examples
            --------
            >>> calculator = InvariantCalculator.calculate_invariants_from_step(Path('path/to/your/step_file.stp'))

            Notes
            -----
            This method assumes the STEP file is well-formed and that the gmsh library is properly installed and configured.
        """
        
        mesh = StepMesher.mesh_from_step(path)
        cls = cls(mesh.path_to_msh)
        cls._process_file()

        return cls
    
    def _process_file(self, normalized=False):
        """
            This function reads a mesh file, applies PCA transformation to the mesh points, optionally normalizes the points,
            and then computes moments and invariants (mues and pies) for the mesh. The results are stored in the instance
            variables `self.mues` and `self.pis`.

            Parameters:
            -----------
            normalized : bool, optional
                If True, the mesh points are scaled to fit within a unit cube. Default is False.
            Raises:
            -------
            Exception
                If there is an error in reading the mesh file or during the computation of moments and invariants.
        """
    
        try:

            inmsh = meshio.read(self.path); 
            tetras_idxs = inmsh.cells_dict['tetra']


            pca = PCA(n_components = 3);    # initialize pca
            pca_fit = pca.fit(inmsh.points);

            inmsh.points = pca.transform(inmsh.points);

            if normalized:      
                max_expension = (np.max(inmsh.points[:,0])-np.min(inmsh.points[:,0])) #get the longest expension
                inmsh.points = inmsh.points/max_expension #scale to unit cube

            tetras = inmsh.points[tetras_idxs]
            tetras_s = np.stack(tetras, axis=-2)  

            scheme = quadpy.t3.get_good_scheme(4)           # initialize integrater 
            row_file_name= self.path.name

            # calculate mues
            for mp in self.moment_permutations:
                    moment_permut_key = f'mue_{mp[0]}{mp[1]}{mp[2]}'
                    l = lambda x: x[0]**mp[0]*x[1]**mp[1]*x[2]**mp[2]    # calculate mues
                    self.mues[moment_permut_key] = scheme.integrate(l,   tetras_s).sum() # integrate over tetrahedrons
            
            # calculate pies
            mue_200 = self.mues['mue_200'] if self.mues['mue_200'] > 0 else self.eps
            mue_020 = self.mues['mue_020'] if self.mues['mue_020'] > 0 else self.eps
            mue_002 = self.mues['mue_002'] if self.mues['mue_002'] > 0 else self.eps

            for mp in self.moment_permutations:
                p=int(mp[0])
                q=int(mp[1])
                r=int(mp[2])

                mue_permut_key = f'mue_{p}{q}{r}'
                pi_permut_key = f'pi_{p}{q}{r}'

                mue_var = self.mues[mue_permut_key]

                self.pis[pi_permut_key] = (mue_var/(mue_200**((4*p-q-r+2)/10)*mue_020**((4*q-p-r+2)/10)*mue_002**((4*r-q-p+2)/10)))

        except Exception as e:
            logger.warning(f"Error processing {self.path.name}: {e}")

    def to_json(self, path : Path):
        """
            Serializes the invariant data to a JSON file.

            This method combines the `mues` and `pis` attributes into a single dictionary
            and writes it to a JSON file at the specified path. The target file will have
            a `.json` extension. If the directory for the target file does not exist, it
            will be created.

            Args:
                path (Path): The file path where the JSON file will be saved.

            Raises:
                ValueError: If the length of the combined `mues` and `pis` dictionary does
                            not match twice the length of `moment_permutations`. In this case, the
                            invariant data may not have been calculated properly.
        """

        self.inv_data = {**self.mues, **self.pis}

        if self.inv_data.__len__() == self.moment_permutations.__len__()*2:
            target_file = path.with_suffix(".json")

            target_file.parent.mkdir(parents=True, exist_ok=True)

            with open(path.with_suffix(".json"), 'w') as f:
                json.dump(self.inv_data, f, indent=4)
