# # Case study: Organocatalytic Mannich Reactions (*JACS* **2023**)

import sys

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
import warnings
from joblib import cpu_count, Parallel, delayed

sys.path.append('../src/')
import RaRFRegressor
import utils


num_jobs = cpu_count()
print(f"Running job with {num_jobs} CPUS")   
np.int = int # fix for skopt

print("Reading data...", flush=True)
df = pd.read_excel('../data/JACS_2023.xlsx')
print("Data read", flush=True)

print("Calculating descriptors...", flush=True)
imine_list = []
for imine in df['Imine_SMILES']:
    imine_list.append(np.array(utils.smi2morgan(imine)))

nu_list = []
for nu in df['pro_nucleophile_SMILES']:
    nu_list.append(np.array(utils.smi2morgan(nu)))

cat_list = []
for cat in df['Catalyst']:
    cat_list.append(np.array(utils.smi2morgan(cat)))

descriptors = np.hstack((imine_list,nu_list,cat_list))
reduced_descriptors = VarianceThreshold().fit_transform(descriptors)

df = pd.DataFrame(data={'refs':df['Reference'].values,'DDG':df['ddG'].values}).join(pd.DataFrame(reduced_descriptors))

print("Descriptors calculated", flush=True)


def run_test(ref):

    print(f"Running test for reference {ref}...", flush=True)

    df_train = df[df['refs']!=ref]
    df_test = df[df['refs']==ref]

    X_train = df_train.iloc[:,2:].values
    X_test = df_test.iloc[:,2:].values
    y_train = df_train['DDG'].values
    y_test = df_test['DDG'].values


    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        min_control = utils.full_model_training_eval(X_train, y_train, X_test, 
                                                     y_test, n_calls=25, show_plot=False)

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
                                              'Reference': [ref]*len(y_test_temp),
                                              'radius': [i]*len(y_test_temp)})
        
        print(result_df, flush=True)
        results.append(result_df)
    
    return pd.concat(results)

all_results = Parallel(n_jobs=num_jobs)(delayed(run_test)(ref) for ref in df['refs'].unique())

combined_results = pd.concat(all_results)

print("Finished", flush=True)
