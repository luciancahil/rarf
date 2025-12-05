import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
import test
import sys
sys.path.append('../../src/')
import utils


df = pd.read_excel('../../data/JACS_2023.xlsx')

imine_list = []
for imine in df['Imine_SMILES']:
    imine_list.append(np.array(utils.smi2morgan(imine)))

nu_list = []
for nu in df['pro_nucleophile_SMILES']:
    nu_list.append(np.array(utils.smi2morgan(nu)))

cat_list = []
for cat in df['Catalyst']:
    cat_list.append(np.array(utils.smi2morgan(cat)))

descriptors = np.hstack((imine_list,nu_list,cat_list))
reduced_df = VarianceThreshold().fit_transform(np.column_stack((df['ddG'].values,descriptors)))
rarf_list, nan_list, control_list, radii_list = test.run_test(10,reduced_df, hp_calls=50)

print(pd.DataFrame({'RARF':rarf_list, 'NaN':nan_list, 'Control':control_list, 'Radii':radii_list}))

