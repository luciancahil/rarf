
# # Case study: Olefin Hydrogenation (*Angew. Chem. Int. Ed.* **2021**)

import sys

from joblib import cpu_count
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

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

X_train, X_test, y_train, y_test = train_test_split(reduced_reactions[:,1:],ddG.values,train_size=0.8, random_state=25)


## RaRF Regression

print("Calculating distances...", flush=True)
train_distances = utils.get_distances(X_train,X_train)
distances = utils.get_distances(X_train,X_test)
print("Distances calculated", flush=True)

RaRF_mae = []
nans_list = []

for i in [0.7,0.8,0.9]:

    print(f"Performing RaRF with radius {i}...", flush=True)

    radius_pred, train_neighbours = RaRFRegressor.RaRFRegressor(radius=i, metric='jaccard').train_parallel(X_train,y_train, 
                                                                                                           include_self='True',
                                                                                                            distances=train_distances,
                                                                                                            n_jobs = num_jobs)
    radius_testpred, test_neighbours = RaRFRegressor.RaRFRegressor(radius=i,metric='jaccard').predict_parallel(X_train, y_train, X_test, distances, n_jobs=num_jobs) 

    nan_indexes = np.where(np.isnan(radius_testpred))[0]
    radius_testpred_temp = np.delete(radius_testpred, nan_indexes)
    y_test_temp = np.delete(y_test, nan_indexes)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.plot(y_train, y_train, color='grey', zorder=0)
    ax1.scatter(y_train, radius_pred, label='train R2 ' + str(round(r2_score(y_train, radius_pred), 2)), color='#279383')
    ax1.scatter(y_test_temp, radius_testpred_temp, label='test R2 ' + str(round(r2_score(y_test_temp, radius_testpred_temp), 2)), color='white', edgecolor='#279383')

    ax1.set_xlabel('Measured $\Delta\Delta G^‡$ (kcal/mol)')
    ax1.set_ylabel('Predicted $\Delta\Delta G^‡$ (kcal/mol)')
    ax1.legend()

    ax2 = sns.kdeplot(data=[train_neighbours, test_neighbours], palette=[COLORA, COLORB])
    ax2.legend(['train', 'test'])
    ax2.set_xlim(-10, len(y_train))
    ax2.set_xlabel('# of neighbours')

    fig.suptitle(f'Radius {i}, {len(nan_indexes)}/{len(radius_testpred)} NaNs')
    plt.tight_layout()
    plt.savefig(f'../results/Olefin_hydrog_RaRF_{i}.png')
    plt.clf()

    mae = mean_absolute_error(y_test_temp, radius_testpred_temp)
    nans_count = len(nan_indexes)
    print("=====================================", flush=True)
    print(f"RaRF with radius {i} done", flush=True)
    print(f"MAE: {mae}, NaNs: {nans_count}", flush=True)
    print("=====================================", flush=True)

    RaRF_mae.append(mae)
    nans_list.append(nans_count)






