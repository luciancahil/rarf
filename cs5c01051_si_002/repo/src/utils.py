import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skopt import gp_minimize
from skopt.space import Categorical, Integer, Real
from skopt.utils import use_named_args

np.int = int # fix for skopt


def smi2morgan(smiles, nbits=2048):
    """
    Convert SMILES to Morgan fingerprint

    Parameters
    ----------
    smiles: SMILES string
    """

    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=nbits)
    mol = Chem.MolFromSmiles(smiles)
    fp = fpgen.GetFingerprint(mol)
    return(fp)

def canonicalize_smiles(smiles):
    """
    Convert SMILES to canonical SMILES

    Parameters
    ----------
    smiles: SMILES string
    """

    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol)

def get_neighbours(X_test, distances, radius):
    """
    get number of neighbours for each reaction in X_test from X_train

    Parameters
    ----------
    X_train: train array of reactions
    X_test: test array of reaction
    distances: precomputed distances between X_train and X_test
    radius: radius for neighbour search
    """
    
    neighbours = []

    for index, rxn in enumerate(X_test):

        indexes = np.where(distances[index, :] <= radius)[0]

        neighbours.append(len(indexes))

    return neighbours

def get_distances(X_train, X_test, metric='jaccard'):
    """
    get distances between each reaction in X_test and X_train using cdist
    
    used to keep conventions consistent (i.e. X_train then X_test)

    Parameters
    ----------
    X_train: array of training reactions
    X_test: array of test reactions
    metric: metric to use for distance calculation 
    """
    X_train = np.array(X_train).astype(float)
    X_test = np.array(X_test).astype(float)
    
    distances = cdist(X_test, X_train, metric=metric)

    return(distances)

def ee2ddg(ee, temp):
    """
    convert enantiomeric excess to free energy difference.

    Parameters
    ----------
    ee: enantiomeric excess
    temp: temperature
    """
    ddg = abs(-0.001986*temp*np.log((100+ee)/(100-ee)))

    return ddg

def ddg2ee(ddg, temp):
    """
    Convert ddg to ee

    Parameter
    ----------
    ddg: Delta delta G value
    temp: temperature
    """
    er = np.exp(-ddg/(0.001986*temp)) 
    ee = (100*(1-er))/(1+er)

    return ee
 
def remove_metal_bonds(smiles, metal='Fe'):
    """
    Remove bonds to metal atoms from a molecule

    Parameter
    ----------
    smiles: SMILES string of molecule
    metal: metal atom to remove
    """

    mol = Chem.MolFromSmiles(smiles, sanitize=False)

    start_remove =[]
    end_remove =[]
    for bond in mol.GetBonds():
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        if (atom1.GetSymbol() == metal) or atom2.GetSymbol() == metal:
            start_remove.append(atom1.GetIdx())
            end_remove.append(atom2.GetIdx())

    mw = Chem.RWMol(mol)
    for start, end in zip(start_remove, end_remove):
        mw.RemoveBond(start, end)
    simplified_smiles = Chem.MolToSmiles(mw)

    return simplified_smiles

def full_model_training_eval(X_train, y_train, X_test, y_test, COLORA="#027F80", n_calls=100,
                             show_plot=True, n_jobs = 1):
    """
    This function trains and evaluates the Random Forest, Neural Network and KNN models on the dataset.
    The hyperparameters are optimized using Bayesian optimization.
    The function returns the minimum MAE of the models.

    Parameter
    ----------
    X_train: array of training reactions
    y_train: array of training labels
    X_test: array of test reactions
    y_test: array of test labels
    COLORA: color for plotting
    n_calls: number of calls for Bayesian optimization
    show_plot: whether to show the plot or not
    n_jobs: number of jobs for parallel processing. Default = 1
    """

    print("Running RF...")
    
    rfpipe = Pipeline([('scaler', StandardScaler()), ('rf', RandomForestRegressor())])

    space  = [Integer(10,200, name='rf__n_estimators'),
              Integer(2,100, name='rf__max_depth'),
              Integer(2,20, name='rf__min_samples_split')]

    @use_named_args(space)
    def objective_rf(**params):
        rfpipe.set_params(**params)
        return -np.mean(cross_val_score(rfpipe, X_train, y_train, cv=5, n_jobs=n_jobs, scoring="neg_mean_absolute_error"))

    rf_gp_min = gp_minimize(objective_rf, space, n_calls=n_calls, random_state=25, verbose=False)

    
    print('=====================')
    print(f'Best hyperparameters are: rf__n_estimators = {rf_gp_min.x[0]},rf__max_depth = {rf_gp_min.x[1]}, rf__min_samples_split = {rf_gp_min.x[2]}')

    rftrain_pred = rfpipe.set_params(rf__n_estimators=rf_gp_min.x[0], rf__max_depth=rf_gp_min.x[1], rf__min_samples_split=rf_gp_min.x[2]).fit(X_train, y_train).predict(X_train)
    rftest_pred = rfpipe.set_params(rf__n_estimators=rf_gp_min.x[0], rf__max_depth=rf_gp_min.x[1], rf__min_samples_split=rf_gp_min.x[2]).fit(X_train, y_train).predict(X_test)
    print("RF Complete")

    print("Running NN...")
    nnpipe = Pipeline([('scaler', StandardScaler()), ('nn', MLPRegressor(max_iter=1000))])

    space = [Integer(10,200, name='nn__hidden_layer_sizes'),
             Categorical(['relu', 'tanh', 'logistic'], name='nn__activation'),
             Real(0.00001,0.001, name='nn__alpha')]

    @use_named_args(space)
    def objective_nn(**params):
        nnpipe.set_params(**params)
        return -np.mean(cross_val_score(nnpipe, X_train, y_train, cv=5, n_jobs=n_jobs, scoring="neg_mean_absolute_error"))


    nn_gp_min = gp_minimize(objective_nn, space, n_calls=n_calls, random_state=25, verbose=False)

    print('=====================')
    print(f'Best hyperparameters are: nn__hidden_layer_sizes = {nn_gp_min.x[0]},nn__activation = {nn_gp_min.x[1]}, nn__alpha = {nn_gp_min.x[2]}')

    mlptrain_pred = nnpipe.set_params(nn__hidden_layer_sizes=nn_gp_min.x[0], nn__activation=nn_gp_min.x[1], nn__alpha=nn_gp_min.x[2]).fit(X_train, y_train).predict(X_train)
    mlptest_pred = nnpipe.set_params(nn__hidden_layer_sizes=nn_gp_min.x[0], nn__activation=nn_gp_min.x[1], nn__alpha=nn_gp_min.x[2]).fit(X_train, y_train).predict(X_test)
    print("NN Complete")

    print("Running KNN...")

    knnpipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsRegressor())])

    space  = [Categorical(['uniform','distance'], name='knn__weights'),
              Categorical(['auto', 'ball_tree', 'kd_tree', 'brute'], name='knn__algorithm'),
              Integer(1,60, name='knn__leaf_size'),
              Integer(1,50, name='knn__n_neighbors'),
              Integer(1,2, name='knn__p')]

    @use_named_args(space)
    def objective_knn(**params):
        knnpipe.set_params(**params)
        return -np.mean(cross_val_score(knnpipe, X_train, y_train, cv=5, n_jobs=n_jobs, scoring="neg_mean_absolute_error"))


    knn_gp_min = gp_minimize(objective_knn, space, n_calls=n_calls, random_state=25, verbose=False)


    print('=====================')
    print(f'Best hyperparameters are: knn__weights = {knn_gp_min.x[0]},knn__algorithm = {knn_gp_min.x[1]}, knn__leaf_size = {knn_gp_min.x[2]}, knn__n_neighbors = {knn_gp_min.x[3]}, knn__p = {knn_gp_min.x[4]}')

    knntrain_pred = knnpipe.set_params(knn__weights=knn_gp_min.x[0], knn__algorithm=knn_gp_min.x[1], knn__leaf_size=knn_gp_min.x[2], knn__n_neighbors=knn_gp_min.x[3], knn__p=knn_gp_min.x[4]).fit(X_train, y_train).predict(X_train)
    knntest_pred = knnpipe.set_params(knn__weights=knn_gp_min.x[0], knn__algorithm=knn_gp_min.x[1], knn__leaf_size=knn_gp_min.x[2], knn__n_neighbors=knn_gp_min.x[3], knn__p=knn_gp_min.x[4]).fit(X_train, y_train).predict(X_test)
    print("KNN Complete")

    def_rfpipe = Pipeline([('scaler', StandardScaler()), ('rf', RandomForestRegressor())])
    def_nnpipe = Pipeline([('scaler', StandardScaler()), ('nn', MLPRegressor(max_iter=1000))])
    def_knnpipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsRegressor())])

    def_rftest_pred = def_rfpipe.fit(X_train, y_train).predict(X_test)
    def_mlptest_pred = def_nnpipe.fit(X_train, y_train).predict(X_test)
    def_knntest_pred = def_knnpipe.fit(X_train, y_train).predict(X_test)

    print('=====================')
    print('R2 score for Random Forest:', r2_score(y_test, rftest_pred))
    print('R2 score for Neural Network:', r2_score(y_test, mlptest_pred))
    print('R2 score for KNN:', r2_score(y_test, knntest_pred))
    print('=====================')
    print('MAE for Random Forest:', mean_absolute_error(y_test, rftest_pred))
    print('MAE for Neural Network:', mean_absolute_error(y_test, mlptest_pred))
    print('MAE for KNN:', mean_absolute_error(y_test, knntest_pred))
    print('=====================')
    print('R2 score for Default Random Forest:', r2_score(y_test, def_rftest_pred))
    print('R2 score for Default Neural Network:', r2_score(y_test, def_mlptest_pred))
    print('R2 score for Default KNN:', r2_score(y_test, def_knntest_pred))
    print('=====================')
    print('MAE for Default Random Forest:', mean_absolute_error(y_test, def_rftest_pred))
    print('MAE for Default Neural Network:', mean_absolute_error(y_test, def_mlptest_pred))
    print('MAE for Default KNN:', mean_absolute_error(y_test, def_knntest_pred))
    print('=====================')

    performance_list = [mean_absolute_error(y_test, rftest_pred), mean_absolute_error(y_test, mlptest_pred), mean_absolute_error(y_test, knntest_pred), 
                        mean_absolute_error(y_test, def_rftest_pred), mean_absolute_error(y_test, def_mlptest_pred), mean_absolute_error(y_test, def_knntest_pred)]

    if show_plot:
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='grey', linestyle='--', zorder=0)
        ax[0].scatter(y_train, rftrain_pred, label='Train', color=COLORA)
        ax[0].scatter(y_test, rftest_pred, label='Test', color='white', edgecolor=COLORA)
        ax[0].set_title('Random Forest')
        ax[0].set_xlabel(r'Measured $\Delta\Delta G^{\ddagger}$ (kcal/mol)')
        ax[0].set_ylabel(r'Predicted $\Delta\Delta G^{\ddagger}$ (kcal/mol)')
        ax[0].text(0.05, 0.95,
                f'R² = {round(r2_score(y_train, rftrain_pred), 2)}\nMAE = {round(mean_absolute_error(y_train, rftrain_pred), 2)}\nQ² = {round(r2_score(y_test, rftest_pred), 2)}\ntest MAE = {round(mean_absolute_error(y_test, rftest_pred), 2)}',
                transform=ax[0].transAxes, verticalalignment='top')
        ax[0].legend()

        ax[1].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='grey', linestyle='--', zorder=0)
        ax[1].scatter(y_train, mlptrain_pred, label='Train', color=COLORA)
        ax[1].scatter(y_test, mlptest_pred, label='Test', color='white', edgecolor=COLORA)
        ax[1].set_title('Neural Network')
        ax[1].set_xlabel(r'Measured $\Delta\Delta G^{\ddagger}$ (kcal/mol)')
        ax[1].set_ylabel(r'Predicted $\Delta\Delta G^{\ddagger}$ (kcal/mol)')
        ax[1].text(0.05, 0.95,
                f'R² = {round(r2_score(y_train, mlptrain_pred), 2)}\nMAE = {round(mean_absolute_error(y_train, mlptrain_pred), 2)}\nQ² = {round(r2_score(y_test, mlptest_pred), 2)}\ntest MAE = {round(mean_absolute_error(y_test, mlptest_pred), 2)}',
                transform=ax[1].transAxes, verticalalignment='top')
        ax[1].legend()

        ax[2].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='grey', linestyle='--', zorder=0)
        ax[2].scatter(y_train, knntrain_pred, label='Train', color=COLORA)
        ax[2].scatter(y_test, knntest_pred, label='Test', color='white', edgecolor=COLORA)
        ax[2].set_title('KNN')
        ax[2].set_xlabel(r'Measured $\Delta\Delta G^{\ddagger}$ (kcal/mol)')
        ax[2].set_ylabel(r'Predicted $\Delta\Delta G^{\ddagger}$ (kcal/mol)')
        ax[2].text(0.05, 0.95,
                f'R² = {round(r2_score(y_train, knntrain_pred), 2)}\nMAE = {round(mean_absolute_error(y_train, knntrain_pred), 2)}\nQ² = {round(r2_score(y_test, knntest_pred), 2)}\ntest MAE = {round(mean_absolute_error(y_test, knntest_pred), 2)}',
                transform=ax[2].transAxes, verticalalignment='top')
        ax[2].legend()

        plt.tight_layout()
        plt.show()

    return min(performance_list)

def full_rf_training_eval(X_train, y_train, X_test, y_test, n_calls=100, n_jobs = 1):
    """
    This function trains and evaluates Random Forest models on the dataset.
    The hyperparameters are optimized using Bayesian optimization.
    The function returns the minimum MAE of the models.

    X_train: array of training reactions
    y_train: array of training labels
    X_test: array of test reactions
    y_test: array of test labels
    n_calls: number of calls for Bayesian optimization
    n_jobs: number of jobs for parallel processing. Default = 1
    """

    print("Running RF...")
    
    rfpipe = Pipeline([('scaler', StandardScaler()), ('rf', RandomForestRegressor())])

    space  = [Integer(10,200, name='rf__n_estimators'),
              Integer(2,100, name='rf__max_depth'),
              Integer(2,20, name='rf__min_samples_split')]

    @use_named_args(space)
    def objective_rf(**params):
        rfpipe.set_params(**params)
        return -np.mean(cross_val_score(rfpipe, X_train, y_train, cv=5, n_jobs=n_jobs, scoring="neg_mean_absolute_error"))

    rf_gp_min = gp_minimize(objective_rf, space, n_calls=n_calls, random_state=25, verbose=False)

    
    print('=====================')
    print(f'Best hyperparameters are: rf__n_estimators = {rf_gp_min.x[0]},rf__max_depth = {rf_gp_min.x[1]}, rf__min_samples_split = {rf_gp_min.x[2]}')

    rftest_pred = rfpipe.set_params(rf__n_estimators=rf_gp_min.x[0], rf__max_depth=rf_gp_min.x[1], rf__min_samples_split=rf_gp_min.x[2]).fit(X_train, y_train).predict(X_test)
    print("RF Complete")

    def_rfpipe = Pipeline([('scaler', StandardScaler()), ('rf', RandomForestRegressor())])
    def_rftest_pred = def_rfpipe.fit(X_train, y_train).predict(X_test)
    print('R2 score for Random Forest:', r2_score(y_test, rftest_pred))
    print('R2 score for Default Random Forest:', r2_score(y_test, def_rftest_pred))

    performance_list = [mean_absolute_error(y_test, rftest_pred), mean_absolute_error(y_test, def_rftest_pred)]
    return min(performance_list)