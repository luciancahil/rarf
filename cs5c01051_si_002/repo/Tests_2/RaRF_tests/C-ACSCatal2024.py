import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
import test
import sys
sys.path.append('../../src/')
import utils


# uploading data
df1 = pd.read_excel('../../data/ACS_Catal_2024.xlsx').dropna()

# getting fingerprints of substances
electrophiles_raw = df1['input electrophile SMILES']
electrophiles = [utils.smi2morgan(electrophiles_raw[i]) for i in range(np.size(electrophiles_raw))]

nucleophiles_raw = df1['nucleophile SMILES']
nucleophiles = [utils.smi2morgan(nucleophiles_raw[i]) for i in range(np.size(nucleophiles_raw))]

substituents3_raw = df1['3,3 Catalyst Substituent']
substituents3 = [utils.smi2morgan(substituents3_raw[i]) for i in range(np.size(substituents3_raw))]

substituentsN_raw = df1['N Catalyst Substituent']
substituentsN = [utils.smi2morgan(substituentsN_raw[i]) for i in range(np.size(substituentsN_raw))]

ddG = df1['ddG']

reactions = np.column_stack((ddG, electrophiles, nucleophiles, substituents3, substituentsN))

reduced_df = VarianceThreshold().fit_transform(reactions)

rarf_list, nan_list, control_list, radii_list = test.run_test(10,reduced_df, hp_calls=50)

print(pd.DataFrame({'RARF':rarf_list, 'NaN':nan_list, 'Control':control_list, 'Radii':radii_list}))

