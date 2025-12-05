
import numpy as np
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.ensemble import RandomForestRegressor
import itertools
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

class kReduction:
    def __init__(self, *, k):
        self.k = k
    
    def cluster_pred(self, X_train, y_train, X_test, y_test):
        """
        Performs kMeans clustering and trains a RF model on each cluster

        Parameters
        ----------
        X_train : np.array
            Training data
        y_train : np.array
            Training labels
        X_test : np.array
            Test data
        y_test : np.array
            Test labels
        """


        clusterer = KMeans(n_clusters=self.k).fit(X_train)
        labels = clusterer.predict(X_train)
        test_labels = clusterer.predict(X_test)

        train_predicted = []
        train_measured = []
        test_measured = []
        test_predicted = []

        for cluster in np.unique(labels):
            index = labels==cluster
            X_train_red = X_train[index]
            y_train_red = y_train[index]

            model = RandomForestRegressor().fit(X_train_red, y_train_red)

            t_index = test_labels == cluster
            X_test_red = X_test[t_index]
            y_test_red = y_test[t_index]

            train_predicted.append(model.predict(X_train_red))
            test_predicted.append(model.predict(X_test_red))

            train_measured.append(y_train_red)
            test_measured.append(y_test_red)
            

        train_measured = list(itertools.chain.from_iterable(train_measured))
        train_predicted = list(itertools.chain.from_iterable(train_predicted))
        test_measured = list(itertools.chain.from_iterable(test_measured))
        test_predicted = list(itertools.chain.from_iterable(test_predicted))

        return(train_measured, train_predicted, test_measured, test_predicted)

    def reduce_pred(self, X_train, y_train, X_test, y_test):
        """
        Perform kReduction -- reduce training set only to k medoids
        and train a RF model on the reduced training set

        Parameters
        ----------
        X_train : np.array
            Training data
        y_train : np.array
            Training labels
        X_test : np.array
            Test data
        y_test : None
        """

        X_train = np.ascontiguousarray(X_train)
        X_test = np.ascontiguousarray(X_test)
        clusterer = KMedoids(n_clusters=self.k).fit(X_train)

        index_list = []
        for center in clusterer.cluster_centers_:
            center_indices = np.where((X_train == center).all(axis=1))[0]
            index_list.extend(center_indices)
        index_list = np.unique(index_list)
        
        X_train_red = X_train[index_list]
        y_train_red = y_train[index_list]
        model = RandomForestRegressor().fit(X_train_red,y_train_red)

        trainmeas = y_train_red
        trainpred = model.predict(X_train_red)

        testpred = model.predict(X_test)


        return(trainmeas, trainpred, testpred)
