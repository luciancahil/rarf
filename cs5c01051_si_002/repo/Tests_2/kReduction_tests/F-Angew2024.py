import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
import test



# uploading data

df = pd.read_csv('../../data/Angew_2024.csv')

reduced_df = VarianceThreshold().fit_transform(np.column_stack((df['ddG'].values,df.iloc[:, 6:])))


kRed = test.run_test(10,reduced_df)

print(pd.DataFrame(kRed))