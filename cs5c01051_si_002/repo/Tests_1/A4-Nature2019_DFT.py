import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
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


df = pd.read_excel('../data/Nature_DFT.xlsx')

reduced_df = VarianceThreshold().fit_transform(df.iloc[:,4:])
df_train, df_test = train_test_split(reduced_df,train_size=0.8, random_state=25)

X_train = df_train[:,1:]
X_test = df_test[:,1:]

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

y_train = df_train[:,0]
y_test = df_test[:,0]

## Controls
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    min_control = utils.full_model_training_eval(X_train, y_train, X_test, 
                                                    y_test, n_calls=100, show_plot=True)

RaRF_mae = []
nans = []
avg_neighbours = []

train_distances = utils.get_distances(X_train,X_train, metric='euclidean')
distances = utils.get_distances(X_train,X_test, metric='euclidean')

max_distance = max(distances.flatten())

for i in np.ceil(np.linspace(0,max_distance,10)):

    radius_pred, train_neighbours = RaRFRegressor.RaRFRegressor(radius=i).train_parallel(X_train,y_train, include_self='True', distances=train_distances)
    radius_testpred, test_neighbours = RaRFRegressor.RaRFRegressor(radius=i).predict_parallel(X_train, y_train, X_test, distances)    
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
    ax1.scatter(y_train,radius_pred, 
                label='train R2 ' + str(round(r2_score(y_train,radius_pred),2)) + ', MAE ' + str(round(mean_absolute_error(y_train, radius_pred),2)), 
                color='#279383')
    ax1.scatter(y_test_temp,radius_testpred_temp, 
                label='test R2 ' + str(round(r2_score(y_test_temp,radius_testpred_temp,),2)) + ', MAE ' + str(round(mean_absolute_error(y_test_temp, radius_testpred_temp),2)), 
                color='white', edgecolor='#279383')

    ax1.set_xlabel('Measured $\Delta\Delta G^‡$ (kcal/mol)')
    ax1.set_ylabel('Predicted $\Delta\Delta G^‡$ (kcal/mol)')
    ax1.legend()


    ax2 = sns.kdeplot(data=[[train_neighbours[x] for x in np.nonzero(train_neighbours)[0]], [test_neighbours[x] for x in np.nonzero(test_neighbours)[0]]], palette=[COLORA, COLORB])
    ax2.legend(['train', 'test'])
    ax2.set_xlim(-10,len(X_train))
    ax2.set_xlabel('# of neighbours')

    fig.suptitle(f'Radius {i}, {len(nan_indexes)}/{len(radius_testpred)} NaNs')
    plt.tight_layout()
    plt.savefig(f'figure_{int(i)}.png')
    plt.close()

    RaRF_mae.append(mean_absolute_error(y_test_temp,radius_testpred_temp))
    
    nans.append(len(nan_indexes))

    avg_neighbours.append(np.average([test_neighbours[x] for x in np.nonzero(test_neighbours)[0]]))

