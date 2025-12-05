import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
import test



# uploading data

df = pd.read_csv('../../data/Angew_2024.csv')

reduced_df = VarianceThreshold().fit_transform(np.column_stack((df['ddG'].values,df.iloc[:, 6:])))


rarf_list, nan_list, control_list, radii_list = test.run_test(10,reduced_df, hp_calls=50)

print(pd.DataFrame({'RARF':rarf_list, 'NaN':nan_list, 'Control':control_list, 'Radii':radii_list}))

