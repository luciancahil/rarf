import sys
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

sys.path.append('../../src/')
import RaRFRegressor
import utils


def run_test(REPS, reduced_df, hp_calls=25, num_jobs=-1):
    """
    Run RF, RaRFRegression.
    """
    
    rarf_list, nan_list, control_list, radii_list = [], [], [], []

    for i in range(REPS):

        print(f'Iteration {i} / {REPS - 1}')
        df_train, df_test = train_test_split(reduced_df, train_size=0.8, random_state=i)
        X_train, X_test = df_train[:, 1:], df_test[:, 1:]
        y_train, y_test = df_train[:, 0], df_test[:, 0]
        
        jaccard_range = np.linspace(0.01, 1.0, 100)
        distances = utils.get_distances(X_train, X_test)
        
        min_control = utils.full_rf_training_eval(X_train, y_train, X_test, y_test, n_calls=hp_calls, n_jobs = num_jobs)

        for radii in jaccard_range:

            radius_testpred, test_neighbours = RaRFRegressor.RaRFRegressor(radius=radii,metric='jaccard').predict_parallel(X_train, y_train, X_test, distances, n_jobs=num_jobs)    
            nan_indexes = []
            index = -1
            for prediction in radius_testpred:
                index +=1
                if np.isnan(prediction) == True:
                    nan_indexes.append(index)
                
            radius_testpred_temp = np.delete(radius_testpred,nan_indexes)
            y_test_temp = np.delete(y_test,nan_indexes)
            error = mean_absolute_error(y_test_temp, radius_testpred_temp) if len(y_test_temp) > 0 else np.nan
            rarf_list.append(error)
            nan_list.append(len(nan_indexes))
            control_list.append(min_control)
            radii_list.append(radii)

    return rarf_list, nan_list, control_list, radii_list
