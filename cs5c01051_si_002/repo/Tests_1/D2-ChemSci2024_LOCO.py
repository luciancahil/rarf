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
df1 = pd.read_excel('../data/Chem_Sci_2024.xlsx', sheet_name=None)
df2 = pd.read_excel('../data/Chem_Sci_2024.xlsx', sheet_name='SM_SMILES')
df3 = df1['pivoted'].iloc[:, [0,1,6]].dropna()

ddG = df3['ddG']

starting_materials = {df2['substrate_name'][i]: utils.smi2morgan(df2['substrate_smiles'][i])
                      for i in range(np.size(df2['substrate_name']))}

starting_materials_reactions = []
for sm in df3['Substrate']:
    starting_materials_reactions.append(starting_materials[sm])

ligands_raw = df1['Ligands'].iloc[:, [0, 6]].dropna()
ligands_raw = ligands_raw.dropna()
converted_smiles = [utils.remove_metal_bonds(lig,'Fe') for lig in ligands_raw['Canonical SMILES']]
ligands = {ligands_raw['Ligand'].iloc[i]: utils.smi2morgan(converted_smiles[i])
           for i in range(np.size(converted_smiles))}

ligands_reactions =[]
for l in df3['Ligand#']:
    ligands_reactions.append(ligands[l])

# fingerprints of reactions

reactions = np.column_stack((ddG, starting_materials_reactions, ligands_reactions))

reduced_reactions = VarianceThreshold().fit_transform(reactions)

df = pd.DataFrame(data={'Catalyst':df3['Ligand#']}).join(pd.DataFrame(reduced_reactions))


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
                                              'Reaction': [cat]*len(y_test_temp),
                                              'radius': [i]*len(y_test_temp)})
        
        print(result_df, flush=True)
        results.append(result_df)
    
    return pd.concat(results)

all_results = Parallel(n_jobs=-1)(delayed(run_test)(ref) for ref in df['Catalyst'].unique())

combined_results = pd.concat(all_results)

print("Finished", flush=True)