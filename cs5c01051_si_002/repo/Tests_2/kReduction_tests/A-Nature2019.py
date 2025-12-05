import pandas as pd
from sklearn.feature_selection import VarianceThreshold
import test


def prepare_dataset(path):

    original_df = pd.read_excel(path, sheet_name='df')
    imines = pd.read_excel(path, sheet_name='imine').set_index('Imine').to_dict(orient='index')
    ligands = pd.read_excel(path, sheet_name='ligand').set_index('ligand').to_dict(orient='index')
    solvents = pd.read_excel(path, sheet_name='solvent').set_index('solvent').to_dict(orient='index')
    nucs = pd.read_excel(path, sheet_name='nuc').set_index('Nucleophile').to_dict(orient='index')

    nuc_descriptors = []
    ligand_descriptors = []
    solvent_descriptors = []
    imine_descriptors = []

    for nuc in original_df['Nucleophile']:
        nuc_descriptors.append(nucs[nuc])

    for ligand in original_df['Ligand']:
        ligand_descriptors.append(ligands[ligand])

    for solvent in original_df['Solvent']:
        solvent_descriptors.append(solvents[solvent])

    for imine in original_df['Imine']:
        imine_descriptors.append(imines[imine])

    descriptors = pd.DataFrame.from_dict(nuc_descriptors).join(pd.DataFrame.from_dict(ligand_descriptors)).join(pd.DataFrame.from_dict(imine_descriptors)).join(pd.DataFrame.from_dict(solvent_descriptors))
    new_df = original_df.join(descriptors)

    return(new_df)

df = prepare_dataset('../../data/Nature_2019.xlsx')
reduced_df = VarianceThreshold().fit_transform(df.iloc[:,6:])

kRed = test.run_test(10,reduced_df)

print(pd.DataFrame(kRed))