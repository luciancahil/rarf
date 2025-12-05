import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
import test
import sys
sys.path.append('../../src/')
import utils

# uploading data
df1 = pd.read_excel('../../data/DD_reactants.xlsx', sheet_name=None)
df2 = pd.read_excel('../../data/DD_data.xlsx', sheet_name='LA-LB-LC-LD').dropna()
df2['ddG'] = utils.ee2ddg(df2[r'%ee'], df2['T (˚C)']+273.15)


# getting fingerprints of substances
ligands_raw = df1['ligands'].dropna()
ligands = {ligands_raw['substrate_name'][i]: utils.smi2morgan(ligands_raw['substrate_smiles'][i])
           for i in range(np.size(ligands_raw['substrate_name']))}

coupling_partners_raw = df1['coupling_partners'].dropna()
coupling_partners = {coupling_partners_raw['substrate_name'][i]: utils.smi2morgan(coupling_partners_raw['substrate_smiles'][i])
                     for i in range(np.size(coupling_partners_raw['substrate_name']))}

substrates_raw = df1['substrates'].dropna()
substrates = {substrates_raw['substrate_name'][i]: utils.smi2morgan(substrates_raw['substrate_smiles'][i])
              for i in range(np.size(substrates_raw['substrate_name']))}

bases_raw = df1['bases'].dropna()
bases = {bases_raw['substrate_name'][i]: utils.smi2morgan(bases_raw['substrate_smiles'][i])
         for i in range(np.size(bases_raw['substrate_name']))}

catalyst_raw = df1['catalyst'].dropna()
catalysts = {catalyst_raw['substrate_name'][i]: utils.smi2morgan(catalyst_raw['substrate_smiles'][i])
            for i in range(np.size(catalyst_raw['substrate_name']))}

solvent_raw = df1['solvent'].dropna()
solvents = {solvent_raw['substrate_name'][i]: utils.smi2morgan(solvent_raw['substrate_smiles'][i])
            for i in range(np.size(solvent_raw['substrate_name']))}

additive_raw = df1['additive'].dropna()
additives = {additive_raw['substrate_name'][i]: utils.smi2morgan(additive_raw['substrate_smiles'][i])
            for i in range(np.size(additive_raw['substrate_name']))}

#fingerprints of reactions

ddG_reactions = df2['ddG']

substrates_reactions =[]
for substrate in df2['substrate']:
    substrates_reactions.append(substrates[substrate])

coupling_partners_reactions =[]
for cp in df2['coupling_partner']:
    coupling_partners_reactions.append(coupling_partners[cp])

catalysts_reactions =[]
for cat in df2['catalyst']:
    catalysts_reactions.append(catalysts[cat])

ligands_reactions = []
for l in df2['ligand']:
    ligands_reactions.append(ligands[l])

bases_reactions = []
for b in df2['base']:
    bases_reactions.append(bases[b])

additives_reactions = []
for ad in df2['additive']:
    additives_reactions.append(additives[ad])

solvents_reactions = []
for solv in df2['solvent']:
    solvents_reactions.append(solvents[solv])

reactions = pd.DataFrame(np.column_stack((ddG_reactions,
                             substrates_reactions, coupling_partners_reactions, catalysts_reactions,
                             ligands_reactions, bases_reactions, additives_reactions, solvents_reactions)))

reduced_df = VarianceThreshold().fit_transform(reactions)

kRed = test.run_test(10,reduced_df)

print(pd.DataFrame(kRed))