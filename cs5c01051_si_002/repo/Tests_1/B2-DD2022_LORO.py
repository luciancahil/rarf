
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

# uploading data
df1 = pd.read_excel('../data/DD_reactants.xlsx', sheet_name=None)
df2 = pd.read_excel('../data/DD_data.xlsx', sheet_name='LA-LB-LC-LD').dropna()
df2['ddG'] = utils.ee2ddg(df2[r'%ee'], df2['T (˚C)']+273.15)


# getting fingerprints of substances
ligands_raw = df1['ligands'].dropna()
ligands = {ligands_raw['substrate_name'][i]: utils.smi2morgan(ligands_raw['substrate_smiles'][i])
           for i in range(np.size(ligands_raw['substrate_name']))}

coupling_partners_raw = df1['coupling_partners'].dropna()
coupling_partners = {coupling_partners_raw['substrate_name'][i]:
                         utils.smi2morgan(coupling_partners_raw['substrate_smiles'][i])
                     for i in range(np.size(coupling_partners_raw['substrate_name']))}

substrates_raw = df1['substrates'].dropna()
substrates = {substrates_raw['substrate_name'][i]:
                  utils.smi2morgan(substrates_raw['substrate_smiles'][i])
              for i in range(np.size(substrates_raw['substrate_name']))}

bases_raw = df1['bases'].dropna()
bases = {bases_raw['substrate_name'][i]:
             utils.smi2morgan(bases_raw['substrate_smiles'][i])
         for i in range(np.size(bases_raw['substrate_name']))}

catalyst_raw = df1['catalyst'].dropna()
catalysts = {catalyst_raw['substrate_name'][i]: utils.smi2morgan(catalyst_raw['substrate_smiles'][i])
            for i in range(np.size(catalyst_raw['substrate_name']))}

solvent_raw = df1['solvent'].dropna()
solvents = {solvent_raw['substrate_name'][i]: utils.smi2morgan(solvent_raw['substrate_smiles'][i])
            for i in range(np.size(solvent_raw['substrate_name']))}

additive_raw = df1['additive'].dropna()
additives = {additive_raw['substrate_name'][i]: utils.smi2morgan(additive_raw['substrate_smiles'][i])
            for i in range(np.size(additive_raw['substrate_name']))}

#fingerprints of reactions

reaction_number = df2['exp_no.']
ddG_reactions = df2['ddG']

substrates_reactions =[]
for substrate in df2['substrate']:
    substrates_reactions.append(substrates[substrate])

coupling_partners_reactions =[]
for cp in df2['coupling_partner']:
    coupling_partners_reactions.append(coupling_partners[cp])

catalysts_reactions =[]
for cat in df2['catalyst']:
    catalysts_reactions.append(catalysts[cat])

ligands_reactions = []
for l in df2['ligand']:
    ligands_reactions.append(ligands[l])

bases_reactions = []
for b in df2['base']:
    bases_reactions.append(bases[b])

additives_reactions = []
for ad in df2['additive']:
    additives_reactions.append(additives[ad])

solvents_reactions = []
for solv in df2['solvent']:
    solvents_reactions.append(solvents[solv])

reactions = pd.DataFrame(np.column_stack((reaction_number, ddG_reactions,
                             substrates_reactions, coupling_partners_reactions, catalysts_reactions,
                             ligands_reactions, bases_reactions, additives_reactions, solvents_reactions)))

reduced_reactions = VarianceThreshold().fit_transform(reactions.iloc[:,1:])

df = pd.DataFrame(data={'refs':df2['exp_no.'].values}).join(pd.DataFrame(reduced_reactions))



def run_test(ref):

    print(f"Running test for reference {ref}...", flush=True)

    df_train = df[df['refs']!=ref]
    df_test = df[df['refs']==ref]

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
                                              'Reference': [ref]*len(y_test_temp),
                                              'radius': [i]*len(y_test_temp)})
        
        print(result_df, flush=True)
        results.append(result_df)
    
    return pd.concat(results)

all_results = Parallel(n_jobs=num_jobs)(delayed(run_test)(ref) for ref in df['refs'].unique())

combined_results = pd.concat(all_results)

print("Finished", flush=True)
