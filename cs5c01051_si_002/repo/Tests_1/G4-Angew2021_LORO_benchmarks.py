
# # Case study: Olefin Hydrogenation (*Angew. Chem. Int. Ed.* **2021**)

import sys

from joblib import Parallel, delayed, cpu_count
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor

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

df = pd.DataFrame(data={'refs':df1['DOI'].values}).join(pd.DataFrame(reduced_reactions))



def run_test(ref):

    print(f"Running test for reference {ref}...", flush=True)

    df_train = df[df['refs']!=ref]
    df_test = df[df['refs']==ref]

    X_train = df_train.iloc[:,2:].values
    X_test = df_test.iloc[:,2:].values
    y_train = df_train.iloc[:,1].values
    y_test = df_test.iloc[:,1].values

    model = RandomForestRegressor(n_estimators=155, max_depth=95, min_samples_split=2).fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results = pd.DataFrame(data={'ref':[ref]*len(y_test), 'y_test':y_test, 'y_pred':y_pred})
    return results

all_results = Parallel(n_jobs=num_jobs)(delayed(run_test)(ref) for ref in df['refs'].unique())

combined_results = pd.concat(all_results)
combined_results.to_csv('../results/OLEFIN_LORO_BENCHMARKS.csv', index=False)

print("Finished", flush=True)




