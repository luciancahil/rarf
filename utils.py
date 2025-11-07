
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np
from scipy.spatial.distance import cdist

def smi2morgan(smi, nbits=2048, radius=2):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        # return zeros if SMILES cannot be parsed
        return np.zeros((nbits,), dtype=int)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    arr = np.zeros((nbits,), dtype=int)  # FIX: allocate full length
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def get_distances(X_train, X_test):
    return cdist(X_test > 0, X_train > 0, metric='jaccard')
