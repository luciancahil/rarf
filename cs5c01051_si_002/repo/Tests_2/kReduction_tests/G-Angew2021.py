import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
import test
import sys
sys.path.append('../../src/')
import utils


# uploading data
df1 = pd.read_csv('../data/Angew_2021.csv')


# getting fingerprints of substances

reactants_raw = df1['ReactantSMILES']
reactants = [utils.smi2morgan(reactants_raw[i]) for i in range(np.size(reactants_raw))]

solvents_raw = df1['SolventSMILES']
solvents = [utils.smi2morgan(solvents_raw[i]) for i in range(np.size(solvents_raw))]

catalysts_raw = '['+df1['Metal']+'].'+df1['Ligand SMILES']
catalysts = [utils.smi2morgan(catalysts_raw[i]) for i in range(np.size(catalysts_raw))]

ddG = df1['ddG']

reactions = np.column_stack((ddG, reactants, solvents, catalysts))

reduced_df = VarianceThreshold().fit_transform(reactions)

kRed = test.run_test(10,reduced_df)

print(pd.DataFrame(kRed))