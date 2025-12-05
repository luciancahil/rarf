import numpy as np
from scipy.spatial.distance import euclidean, jaccard
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm 
from joblib import Parallel, delayed

class RaRFRegressor:
    def __init__(self, *, radius=1, metric='jaccard'):
        self.radius = radius
        if metric=='euclidean':
            self.metric = euclidean
        if metric=='jaccard':
            self.metric = jaccard
    
    def train_predict(self, X_train, y_train, include_self='False', distances=None):
        """
        RaRF regression on a training set

        Parameters
        ----------
        X_train: Array of training parameters
        y_train: Array of training responses
        include_self: For a given reaction, include itself in the training set. If False,
        equivalent to LOO.
        distances: Precomputed distances between training set reactions
        """
        
        predictions = []
        neighbours = []

        print("Training model...")
        for rxn_index,rxn in enumerate(tqdm(X_train)):

            indexes = []

            if distances is not None:

                if include_self:
                    indexes = np.where(distances[rxn_index, :] <= self.radius)[0]
                else:
                    indexes = np.where(distances[rxn_index, :] <= self.radius)[0]
                    indexes = indexes[indexes != rxn_index]
            
            else:
                if include_self:
                    for index,train_rxn in enumerate(X_train):
                        distance = self.metric(rxn,train_rxn)
                        if distance <= self.radius:
                                indexes.append(index)

                else:
                    for index,train_rxn in enumerate(X_train):
                        distance = self.metric(rxn,train_rxn)
                        if distance <= self.radius:
                            if rxn_index != index:
                                indexes.append(index)
            
            X_train_red = X_train[indexes]
            y_train_red = y_train[indexes]

            if len(X_train_red) == 0:
                print("NO NEIGHBOURS DETECTED, PREDICTING nan")
                predictions.append(np.nan)
            elif X_train_red.ndim == 1:
                predictions.append(y_train_red)
            else:
                model = RandomForestRegressor().fit(X_train_red,y_train_red)
                predictions.append(float(model.predict(rxn.reshape(1,-1))))
                
            neighbours.append(len(y_train_red))
        return predictions, neighbours
        
    def predict(self, X_train, y_train, X_test, distances): 
        """
        Train on known values and predict unknown values. Currently only configured for n_neighbours=int.
        
        Parameters
        ----------
        X_train: Array of training parameters
        y_train: Array of training responses
        X_test: Arry of test parameters
        distances: Precomputed distances between test and training set reactions
        """
        predictions = []
        neighbours = []

        print("Predicting values...")
        for test_index, test_rxn in enumerate(tqdm(X_test)):

            indexes = np.where(distances[test_index, :] <= self.radius)[0]
            X_train_red = X_train[indexes]
            y_train_red = y_train[indexes]

            if len(X_train_red) == 0:
                print("NO NEIGHBOURS DETECTED, PREDICTING nan")
                predictions.append(np.nan)
            elif X_train_red.ndim == 1:
                predictions.append(y_train_red)
            else:
                model = RandomForestRegressor().fit(X_train_red,y_train_red)
                predictions.append(float(model.predict(test_rxn.reshape(1,-1))))

            neighbours.append(len(y_train_red))

        return np.array(predictions), neighbours
    
    def predict_parallel(self,X_train, y_train, X_test, distances, n_jobs=-1):
        """ 
        Parallel version of predict function

        Parameters
        ----------
        X_train: Array of training parameters
        y_train: Array of training responses
        X_test: Array of test parameters
        distances: Precomputed distances between training set reactions
        n_jobs: Number of parallel jobs to run
        """

        def process(test_index, test_rxn):
            indexes = np.where(distances[test_index, :] <= self.radius)[0]
            X_train_red = X_train[indexes]
            y_train_red = y_train[indexes]

            if len(X_train_red) == 0:
                return np.nan, len(y_train_red)
            
            elif X_train_red.ndim == 1:
                return y_train_red, len(y_train_red)
            else:
                model = RandomForestRegressor().fit(X_train_red, y_train_red)
                return float(model.predict(test_rxn.reshape(1, -1))), len(y_train_red)

        results = Parallel(n_jobs=n_jobs)(delayed(process)(test_index, test_rxn) for test_index, test_rxn in enumerate(X_test))

        predictions, neighbours = zip(*results)
        predictions = np.array(predictions)
        neighbours = np.array(neighbours)

        return predictions, neighbours
    
    def train_parallel(self, X_train, y_train, include_self='False', distances=None, n_jobs=-1):
        """
        Parallel version of train_predict function

        Parameters
        ----------
        X_train: Array of training parameters
        y_train: Array of training responses
        include_self: For a given reaction, include itself in the training set. If False,
        equivalent to LOO.
        distances: Precomputed distances between training set reactions
        n_jobs: Number of parallel jobs to run
        """
        def process(rxn_index, rxn):
            indexes = []

            if distances is not None:

                if include_self:
                    indexes = np.where(distances[rxn_index, :] <= self.radius)[0]
                else:
                    indexes = np.where(distances[rxn_index, :] <= self.radius)[0]
                    indexes = indexes[indexes != rxn_index]
            
            else:
                if include_self:
                    for index,train_rxn in enumerate(X_train):
                        distance = self.metric(rxn,train_rxn)
                        if distance <= self.radius:
                                indexes.append(index)

                else:
                    for index,train_rxn in enumerate(X_train):
                        distance = self.metric(rxn,train_rxn)
                        if distance <= self.radius:
                            if rxn_index != index:
                                indexes.append(index)
            
            X_train_red = X_train[indexes]
            y_train_red = y_train[indexes]

            if len(X_train_red) == 0:
                return np.nan, len(y_train_red)
            elif X_train_red.ndim == 1:
                return y_train_red, len(y_train_red)
            else:
                model = RandomForestRegressor().fit(X_train_red,y_train_red)
                return float(model.predict(rxn.reshape(1,-1))), len(y_train_red)
        
        results = Parallel(n_jobs=n_jobs)(delayed(process)(rxn_index, rxn) for rxn_index, rxn in enumerate(X_train))

        predictions, neighbours = zip(*results)
        predictions = np.array(predictions)
        neighbours = np.array(neighbours)

        return predictions, neighbours