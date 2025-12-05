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
COLORA = '#027F80'
COLORB = '#B2E5FC'

# uploading data
df1 = pd.read_excel('../data/ACS_Catal_2024.xlsx').dropna()

# getting fingerprints of substances
electrophiles_raw = df1['input electrophile SMILES']
electrophiles = [utils.smi2morgan(electrophiles_raw[i]) for i in range(np.size(electrophiles_raw))]

nucleophiles_raw = df1['nucleophile SMILES']
nucleophiles = [utils.smi2morgan(nucleophiles_raw[i]) for i in range(np.size(nucleophiles_raw))]

substituents3_raw = df1['3,3 Catalyst Substituent']
substituents3 = [utils.smi2morgan(substituents3_raw[i]) for i in range(np.size(substituents3_raw))]

substituentsN_raw = df1['N Catalyst Substituent']
substituentsN = [utils.smi2morgan(substituentsN_raw[i]) for i in range(np.size(substituentsN_raw))]

ddG = df1['ddG']

reactions = np.column_stack((ddG, electrophiles, nucleophiles, substituents3, substituentsN))

reduced_reactions = VarianceThreshold().fit_transform(reactions)

df = pd.DataFrame(data={'Reaction':df1['reference'].values}).join(pd.DataFrame(reduced_reactions))



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