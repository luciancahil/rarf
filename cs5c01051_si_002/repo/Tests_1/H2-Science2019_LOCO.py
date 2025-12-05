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
df = pd.read_excel('../data/Science_2019.xlsx', sheet_name='performance')

ba_list = []
for ba in df['Catalyst']:
    ba_list.append(np.array(utils.smi2morgan(ba)))

imine_list = []
for im in df['Imine']:
    imine_list.append(np.array(utils.smi2morgan(im)))

thiol_list = []
for th in df['Thiol']:
    thiol_list.append(np.array(utils.smi2morgan(th)))


descriptors = np.hstack((ba_list,imine_list,thiol_list))
reduced_reactions = VarianceThreshold().fit_transform(descriptors)

df = pd.DataFrame(data={'Catalyst':df['Catalyst'].values, 'ddG':df['Output'].values}).join(pd.DataFrame(reduced_reactions))

df['Catalyst'] = [utils.canonicalize_smiles(canon_cat) for canon_cat in df['Catalyst']]


def run_test(cat):

    print(f"Running test for catalyst {cat}...", flush=True)

    df_train = df[df['Catalyst']!=cat]
    df_test = df[df['Catalyst']==cat]

    X_train = df_train.iloc[:,2:].values
    X_test = df_test.iloc[:,2:].values
    y_train = df_train.iloc[:,1].values
    y_test = df_test.iloc[:,1].values


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
                                              'Reaction': [cat]*len(y_test_temp),
                                              'radius': [i]*len(y_test_temp)})
        
        results.append(result_df)
    
    return pd.concat(results)

all_results = Parallel(n_jobs=-1)(delayed(run_test)(ref) for ref in df['Catalyst'].unique())

combined_results = pd.concat(all_results)

print("Finished", flush=True)