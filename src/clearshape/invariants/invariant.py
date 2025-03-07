# standard libaries
from pathlib import Path
import os

# Third party libaries
import meshio
import numpy as np
import json
import quadpy
from sklearn.decomposition import PCA 

# application imports
#import clearshape.invariants.mesher as StepMesher
from invariants.mesher import StepMesher
#from clearshape.invariants.mehser as StepMesher


class InvariantCalculator:

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
        """Creates a 'InvariantCalculator' instance from a path to a mesh file.
         
          The method reads the mesh file specified in path and returns the invariants as a numpy array."""
        
        mesh = StepMesher.mesh_from_step(path)
        cls = cls(mesh.path_to_msh)
        cls._process_file()

        return cls
    
    def _process_file(self, normalized=False):

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
            print(f"Error frocessing {self.path.name}: {e}")

    def to_json(self, path : Path):

        self.inv_data = {**self.mues, **self.pis}

        if self.inv_data.__len__() == self.moment_permutations.__len__()*2:
            target_file = path.with_suffix(".json")

            target_file.parent.mkdir(parents=True, exist_ok=True)

            with open(path.with_suffix(".json"), 'w') as f:
                json.dump(self.inv_data, f, indent=4)
