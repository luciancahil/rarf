import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
import test
import sys
sys.path.append('../../src/')
import utils


# uploading data
df1 = pd.read_excel('../../data/Chem_Sci_2024.xlsx', sheet_name=None)
df2 = pd.read_excel('../../data/Chem_Sci_2024.xlsx', sheet_name='SM_SMILES')
df3 = df1['pivoted'].iloc[:, [0,1,6]].dropna()

ddG = df3['ddG']

starting_materials = {df2['substrate_name'][i]: utils.smi2morgan(df2['substrate_smiles'][i])
                      for i in range(np.size(df2['substrate_name']))}

starting_materials_reactions = []
for sm in df3['Substrate']:
    starting_materials_reactions.append(starting_materials[sm])

ligands_raw = df1['Ligands'].iloc[:, [0, 6]].dropna()
ligands_raw = ligands_raw.dropna()
converted_smiles = [utils.remove_metal_bonds(lig,'Fe') for lig in ligands_raw['Canonical SMILES']]
ligands = {ligands_raw['Ligand'].iloc[i]: utils.smi2morgan(converted_smiles[i])
           for i in range(np.size(converted_smiles))}

ligands_reactions =[]
for l in df3['Ligand#']:
    ligands_reactions.append(ligands[l])

# fingerprints of reactions

reactions = np.column_stack((ddG, starting_materials_reactions, ligands_reactions))

reduced_df = VarianceThreshold().fit_transform(reactions)

kRed = test.run_test(10,reduced_df)

print(pd.DataFrame(kRed))