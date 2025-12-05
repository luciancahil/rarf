
# # Case study: Olefin Hydrogenation (*Angew. Chem. Int. Ed.* **2021**)

import sys

from joblib import Parallel, delayed, cpu_count
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
import warnings

sys.path.append('../src/')
import RaRFRegressor
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


    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore", UserWarning)
    #     min_control = utils.full_model_training_eval(X_train, y_train, X_test, 
    #                                                  y_test, n_calls=25, show_plot=False, n_jobs=1)

    print("Calculating distances...", flush=True)
    distances = utils.get_distances(X_train,X_test)
    print("Distances calculated", flush=True)

    results = []
    for i in np.divide(range(1,10),10):
        nebs = utils.get_neighbours(X_test,distances,radius=i)
        result_df = pd.DataFrame({'radius':[i]*len(nebs), 'neighbours':np.divide(nebs,len(X_train))})

        # radius_testpred, test_neighbours = RaRFRegressor.RaRFRegressor(radius=i,metric='jaccard').predict(X_train, y_train, X_test, distances)    

        # nan_indexes = np.where(np.isnan(radius_testpred))[0]
        # radius_testpred_temp = np.delete(radius_testpred,nan_indexes)
        # y_test_temp = np.delete(y_test,nan_indexes)

        # print(f"LENGTHS {len(y_test_temp)}, {len(radius_testpred_temp)}, {len([len(nan_indexes)]*len(y_test_temp))}, {len([ref]*len(y_test_temp))}, {len([i]*len(y_test_temp))}, ", flush=True)

        # result_df = pd.DataFrame({'y_test': y_test_temp, 
        #                                       'y_pred': radius_testpred_temp, 
        #                                       'nans': [len(nan_indexes)]*len(y_test_temp), 
        #                                     #   'control': [min_control]*len(y_test_temp),
        #                                       'Reference': [ref]*len(y_test_temp),
        #                                       'radius': [i]*len(y_test_temp)})
        
        results.append(result_df)

    # pd.concat(results).to_csv(f'../results/olefinhydrog_LORO_{hash(ref)}.csv', index=False)
    
    return pd.concat(results)

all_results = Parallel(n_jobs=num_jobs)(delayed(run_test)(ref) for ref in df['refs'].unique())

combined_results = pd.concat(all_results)
combined_results.to_csv('../results/LORO_NEIGHBOURS.csv', index=False)

print("Finished", flush=True)




