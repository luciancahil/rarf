
# Case study: iDPI reactions (*ACS Catal.* **2024**)

import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import warnings

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

df_train, df_test = train_test_split(reduced_reactions, train_size=0.8, random_state=25)
X_train, X_test = df_train[:, 1:], df_test[:, 1:]
y_train, y_test = df_train[:, 0], df_test[:, 0]
 
## Controls
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    min_control = utils.full_model_training_eval(X_train, y_train, X_test, 
                                                    y_test, n_calls=100, show_plot=True)

## RaRF Regression

RaRF_mae = []
nans = []
avg_neighbours = []

distances = utils.get_distances(X_train,X_test)

for i in np.divide(range(1,10),10):


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


    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5), dpi=300)


    ax1.plot(y_train,y_train, color='grey', zorder=0)
    ax1.scatter(y_train,radius_pred, label='train R2 ' + str(round(r2_score(y_train,radius_pred),2)), color='#279383')
    ax1.scatter(y_test_temp,radius_testpred_temp, label='test R2 ' + str(round(r2_score(y_test_temp,radius_testpred_temp,),2)), color='white', edgecolor='#279383')

    ax1.set_xlabel('Measured $\Delta\Delta G^‡$ (kcal/mol)')
    ax1.set_ylabel('Predicted $\Delta\Delta G^‡$ (kcal/mol)')
    ax1.legend()


    ax2 = sns.kdeplot(data=[[train_neighbours[x] for x in np.nonzero(train_neighbours)[0]], [test_neighbours[x] for x in np.nonzero(test_neighbours)[0]]], palette=[COLORA, COLORB])
    ax2.legend(['train', 'test'])
    ax2.set_xlim(-10,200)
    ax2.set_xlabel('# of neighbours')

    fig.suptitle(f'Radius {i}, {len(nan_indexes)}/{len(radius_testpred)} NaNs')
    plt.tight_layout()
    plt.show()

    RaRF_mae.append(mean_absolute_error(y_test_temp,radius_testpred_temp))
    
    nans.append(len(nan_indexes))

    avg_neighbours.append(np.average([test_neighbours[x] for x in np.nonzero(test_neighbours)[0]]))



