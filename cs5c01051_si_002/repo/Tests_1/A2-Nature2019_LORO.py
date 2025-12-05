import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import warnings
from sklearn.feature_selection import VarianceThreshold
from joblib import Parallel, delayed

sys.path.append('../src/')
import RaRFRegressor
import utils

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14

def prepare_dataset(path):

    original_df = pd.read_excel(path, sheet_name='df')
    imines = pd.read_excel(path, sheet_name='imine').set_index('Imine').to_dict(orient='index')
    ligands = pd.read_excel(path, sheet_name='ligand').set_index('ligand').to_dict(orient='index')
    solvents = pd.read_excel(path, sheet_name='solvent').set_index('solvent').to_dict(orient='index')
    nucs = pd.read_excel(path, sheet_name='nuc').set_index('Nucleophile').to_dict(orient='index')

    nuc_descriptors = []
    ligand_descriptors = []
    solvent_descriptors = []
    imine_descriptors = []

    for nuc in original_df['Nucleophile']:
        nuc_descriptors.append(nucs[nuc])

    for ligand in original_df['Ligand']:
        ligand_descriptors.append(ligands[ligand])

    for solvent in original_df['Solvent']:
        solvent_descriptors.append(solvents[solvent])

    for imine in original_df['Imine']:
        imine_descriptors.append(imines[imine])

    descriptors = pd.DataFrame.from_dict(nuc_descriptors).join(pd.DataFrame.from_dict(ligand_descriptors)).join(pd.DataFrame.from_dict(imine_descriptors)).join(pd.DataFrame.from_dict(solvent_descriptors))
    new_df = original_df.join(descriptors)

    return(new_df)

df = prepare_dataset('../data/Nature_2019.xlsx')
reduced_df = VarianceThreshold().fit_transform(df.iloc[:,6:])

df = pd.DataFrame(data={'Reaction':df['Reaction'].values}).join(pd.DataFrame(reduced_df))



def run_test(ref):

    print(f"Running test for reference {ref}...", flush=True)

    df_train = df[df['Reaction']!=ref]
    df_test = df[df['Reaction']==ref]

    X_train = df_train.iloc[:,2:].values
    X_test = df_test.iloc[:,2:].values
    y_train = df_train.iloc[:,1].values
    y_test = df_test.iloc[:,1].values


    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        min_control = utils.full_model_training_eval(X_train, y_train, X_test, 
                                                     y_test, n_calls=50, show_plot=False)

    print("Calculating distances...", flush=True)
    distances = utils.get_distances(X_train,X_test)
    print("Distances calculated", flush=True)

    results = []
    for i in np.divide(range(1,10),10):

        radius_testpred, test_neighbours = RaRFRegressor.RaRFRegressor(radius=i,metric='jaccard').predict(X_train, y_train, X_test, distances)    

        nan_indexes = np.where(np.isnan(radius_testpred))[0]
        radius_testpred_temp = np.delete(radius_testpred,nan_indexes)
        y_test_temp = np.delete(y_test,nan_indexes)
        
        result_df = pd.DataFrame({'y_test': y_test_temp, 
                                              'y_pred': radius_testpred_temp, 
                                              'nans': [len(nan_indexes)]*len(y_test_temp), 
                                              'control': [min_control]*len(y_test_temp),
                                              'Reaction': [ref]*len(y_test_temp),
                                              'radius': [i]*len(y_test_temp)})
        
        print(result_df, flush=True)
        results.append(result_df)
    
    return pd.concat(results)

all_results = Parallel(n_jobs=-1)(delayed(run_test)(ref) for ref in df['Reaction'].unique())

combined_results = pd.concat(all_results)

print("Finished", flush=True)