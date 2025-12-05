import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
import ast

import sys
sys.path.append('../src/')
import RaRFRegressor
import utils

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14
COLORA = '#027F80'
COLORB = '#B2E5FC'

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

    return(descriptors)

df = prepare_dataset('../data/Nature_SMILES.xlsx')

ddg = pd.read_excel('../data/Nature_2019.xlsx', sheet_name='df').loc[:,'DDG'].values

results = []
for nbits in [512, 1028, 2048, 4096]:
    nu = [np.array(utils.smi2morgan(x, nbits=nbits)) for x in df['SMILES_n']]
    li = [np.array(utils.smi2morgan(x, nbits=nbits)) for x in df['SMILES_l']]
    im = [np.array(utils.smi2morgan(x, nbits=nbits)) for x in df['SMILES_i']]
    so = [np.array(utils.smi2morgan(x, nbits=nbits)) for x in df['SMILES_s']]

    X_full = np.hstack((nu,li,im,so))
    
    X = VarianceThreshold().fit_transform(X_full)
    X_train, X_test, y_train, y_test = train_test_split(X, ddg, train_size=0.8, random_state=25)

    RaRF_mae = []

    distances = utils.get_distances(X_train,X_test)

    for i in np.divide(range(1,10),10):
        print('Radius:',i, 'Bits:',nbits)

        radius_pred, train_neighbours = RaRFRegressor.RaRFRegressor(radius=i, metric='jaccard').train_parallel(X_train,y_train, include_self='True')
        radius_testpred, test_neighbours = RaRFRegressor.RaRFRegressor(radius=i,metric='jaccard').predict_parallel(X_train, y_train, X_test, distances)    
        
        nan_indexes = []
        index = -1
        for prediction in radius_testpred:
            index +=1
            if np.isnan(prediction) == True:
                nan_indexes.append(index)
            
        radius_testpred_temp = np.delete(radius_testpred,nan_indexes)
        y_test_temp = np.delete(y_test,nan_indexes)


        RaRF_mae.append(mean_absolute_error(y_test_temp,radius_testpred_temp))
        
    results.append((nbits, RaRF_mae))

pd.DataFrame(results, columns=['nbits', 'RaRF_mae']).to_csv('../results/Tests1/A-Nature2019_bits.csv', index=False)

# plot results
results = pd.read_csv('../results/Tests1/A-Nature2019_bits.csv')

for bit in results['nbits']:
    maes = ast.literal_eval(results.loc[results['nbits']==bit,'RaRF_mae'].values[0])
    plt.plot(np.divide(range(1,10),10), maes, label=str(bit))

plt.axhline(y=0.27, color='black', linestyle='--', label='Full training set', zorder=0)
plt.legend(frameon=False, loc='lower right')
plt.xlabel('Radius')
plt.ylabel('Mean Absolute Error (kcal/mol)')
# plt.ylim(0,0.5)
plt.show()



