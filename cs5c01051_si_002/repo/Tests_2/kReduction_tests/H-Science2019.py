import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
import test
import sys
sys.path.append('../../src/')
import utils


df = pd.read_excel('../../data/Science_2019.xlsx', sheet_name='performance')

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
reduced_df = VarianceThreshold().fit_transform(np.column_stack((df['Output'].values,descriptors)))

kRed = test.run_test(10,reduced_df)

print(pd.DataFrame(kRed))