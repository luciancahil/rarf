# Repository for "Evaluating Predictive Accuracy in Asymmetric Catalysis: A Machine Learning Perspective on Local Reaction Space"
This repository contains code and data accompanying the manuscript: "Evaluating Predictive Accuracy in Asymmetric Catalysis: A Machine Learning Perspective on Local Reaction Space"

Repository structure:

Data: Raw data files used for modelling. Data for olefin hydrogenation should be directed to Xin Hong at Zhejiang University (hxchem@zju.edu.cn). To reproduce tests for this case study, users should obtain the dataset and rename it to "Angew_2021" inside the data folder. All other data sources are provided herein and obtained from references discussed in the supplementary information.

Tests1: Notebooks and scripts used to reproduce section 1 of the paper. Jupyter notebooks are provided for some smaller case studies and serve as examples to easily reproduce results. Most scripts are provided  directly as python scripts for streamlined submission to HPCs. 

Tests2: Notebooks and scripts used to reproduce section 2 of the paper.

src: Source code for RaRFRegressor, kReduction regressor, and helper functions.

Environment files are included for the users convenience for conda (environment.yml) or pip (requirements.txt).

### Usage

```python
# Example usage of RaRFRegressor within this repository

import sys
sys.path.append('../src/')
import RaRFRegressor
import utils

# Obtain distances first to speed calculations
train_distances = utils.get_distances(X_train,X_train)
distances = utils.get_distances(X_train,X_test)

# Predict training set
radius_pred, train_neighbours = RaRFRegressor.RaRFRegressor(radius=i, metric='jaccard').train_predict(X_train,y_train.values, include_self='True', distances=train_distances) # This function utilizes array indexing and requires inputs to be indexless (i.e. arrays, not pandas dataframes)

# Predict test set
radius_testpred, test_neighbours = RaRFRegressor.RaRFRegressor(radius=i,metric='jaccard').predict(X_train, y_train.values, X_test, distances)

# For large case studies, a parallelized version is included which handles multiple reactions at one time
radius_testpred, test_neighbours = RaRFRegressor.RaRFRegressor(radius=i,metric='jaccard').predict_parallel(X_train, y_train, X_test, distances, n_jobs=-1)    

```