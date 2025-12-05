# # Case study: Olefin Hydrogenation (*Angew. Chem. Int. Ed.* **2021**)

import sys

from joblib import cpu_count
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
import warnings

sys.path.append('../src/')
import utils


plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14
COLORA = '#027F80'
COLORB = '#B2E5FC'

num_jobs = cpu_count()
print(f"Running Job with {num_jobs} CPUs", flush=True)

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

reduced_reactions = VarianceThreshold().fit_transform(reactions)

X_train, X_test, y_train, y_test = train_test_split(reduced_reactions[:,1:],ddG.values,train_size=0.8, random_state=25)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    min_control = utils.full_model_training_eval(X_train, y_train, X_test, 
                                                    y_test, n_calls=50, show_plot=False, n_jobs=-1)