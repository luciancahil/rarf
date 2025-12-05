import sys
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

sys.path.append('../../src/')
from kReduction import kReduction
import utils


def kreduction(i, X_train, y_train, X_test, y_test):
    """
    Apply kReduction to predict test set and calculate error.
    """
    trainmeas, trainpred, testpred = kReduction(k=i).reduce_pred(X_train, y_train, X_test, y_test)
    return mean_absolute_error(y_test, testpred)


def run_test(REPS, reduced_df, num_jobs=-1):
    """
    Run the test for multiple repetitions, only kreduction.
    """
    
    kRed_list = []

    for i in range(REPS):
        print(f'Iteration {i} / {REPS - 1}')
        df_train, df_test = train_test_split(reduced_df, train_size=0.8, random_state=i)
        X_train, X_test = df_train[:, 1:], df_test[:, 1:]
        y_train, y_test = df_train[:, 0], df_test[:, 0]
        
        jaccard_range = np.linspace(0.01, 1.0, 100)
        distances = utils.get_distances(X_train, X_test)

        k_range = []

        for distance in jaccard_range:
            nebs = utils.get_neighbours(X_test,distances,distance)
            try:
                k_range.append(int(np.ceil(np.average([nebs[x] for x in np.nonzero(nebs)[0]]))))
            except:
                k_range.append(1)


        k_range[k_range == 0] = 1

        with Parallel(n_jobs=num_jobs) as parallel:
            kmae_list = parallel(delayed(kreduction)(k, X_train, y_train, X_test, y_test) for k in k_range)

        kRed_list.extend(kmae_list)

    return kRed_list

