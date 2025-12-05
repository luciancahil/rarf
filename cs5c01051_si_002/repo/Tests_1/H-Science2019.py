
#Case study: Thiol additions (*Science* **2019**)


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
reduced_descriptors = VarianceThreshold().fit_transform(descriptors)

X_train, X_test, y_train, y_test = train_test_split(reduced_descriptors,df['Output'].values,train_size=0.8, random_state=25)


# ## Controls


# RF


## Controls
# with warnings.catch_warnings():
#     warnings.simplefilter("ignore", UserWarning)
#     min_control = utils.full_model_training_eval(X_train, y_train, X_test, 
#                                                     y_test, n_calls=50, show_plot=True)


# ## RaRF Regression


RaRF_mae = []
nans = []

train_distances = utils.get_distances(X_train,X_train)
distances = utils.get_distances(X_train,X_test)

for i in np.divide(range(1,10),10):
    radius_pred, train_neighbours = RaRFRegressor.RaRFRegressor(radius=i, metric='jaccard').train_parallel(X_train,y_train, 
                                                                                                           include_self='True',
                                                                                                            distances=train_distances,
                                                                                                            n_jobs = -1)
    radius_testpred, test_neighbours = RaRFRegressor.RaRFRegressor(radius=i,metric='jaccard').predict_parallel(X_train, y_train, X_test, distances, n_jobs=-1) 

    nan_indexes = []
    index = -1
    for prediction in radius_testpred:
        index +=1
        if np.isnan(prediction) == True:
            nan_indexes.append(index)
        
    radius_testpred_temp = np.delete(radius_testpred,nan_indexes)
    y_test_temp = np.delete(y_test,nan_indexes)


    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))


    ax1.plot(y_train,y_train, color='grey', zorder=0)
    ax1.scatter(y_train,radius_pred, label='train R2 ' + str(round(r2_score(y_train,radius_pred),2)), color='#279383')
    ax1.scatter(y_test_temp,radius_testpred_temp, label='test R2 ' + str(round(r2_score(y_test_temp,radius_testpred_temp,),2)), color='white', edgecolor='#279383')

    ax1.set_xlabel('Measured $\Delta\Delta G^‡$ (kcal/mol)')
    ax1.set_ylabel('Predicted $\Delta\Delta G^‡$ (kcal/mol)')
    ax1.legend()


    ax2 = sns.kdeplot(data=[train_neighbours, test_neighbours], palette=[COLORA, COLORB])
    ax2.legend(['train', 'test'])
    
    if i == 0.9:
        ax2.set_xlim(-10,len(y_train))
    else:
        ax2.set_xlim(-10,200)

    ax2.set_xlabel('# of neighbours')

    fig.suptitle(f'Radius {i}, {len(nan_indexes)}/{len(radius_testpred)} NaNs')
    plt.tight_layout()
    plt.show()

    RaRF_mae.append(mean_absolute_error(y_test_temp,radius_testpred_temp))
    
    nans.append(len(nan_indexes))



