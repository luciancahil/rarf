
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from RaRFRegressor_shared_overlap_visual import RaRFRegressor
from utils import smi2morgan
import matplotlib.pyplot as plt

def load_nature_data():
    xlsx = "Nature_SMILES.xlsx"
    df = pd.read_excel(xlsx, sheet_name="df")

    solvent = pd.read_excel(xlsx, sheet_name='solvent')
    imine =  pd.read_excel(xlsx, sheet_name='imine')
    nuc =  pd.read_excel(xlsx, sheet_name='nuc')
    ligand = pd.read_excel(xlsx, sheet_name='ligand')

    solvent = dict(zip(solvent['solvent'], solvent['SMILES_s']))
    imine = dict(zip(imine['Imine'], imine['SMILES_i']))
    nuc = dict(zip(nuc['Nucleophile'], nuc['SMILES_n']))
    ligand = dict(zip(ligand['ligand'], ligand['SMILES_l']))

    smiles_all = []

    for index, row in df.iterrows():
        cur_smiles = [solvent[row["Solvent"]], imine[row["Imine"]], nuc[row["Nucleophile"]], ligand[row["Ligand"]]]

        smiles_all.append(cur_smiles)
    
    y = df["DDG"].values


    return smiles_all[0:366], y[0:366]


def load_upup_data():
    csv_file_path = "database_ml-new-up-up.csv"

    df = pd.read_csv(csv_file_path)

    X_cols = ["Solvent SMILES","Ligand SMILES","New Catalyst SMILES(RDKit)","Reactant SMILES","Product SMILES"]

    y_col = 'ddG'
    

    smiles_all = []
    for index, row in df.iterrows():
        cur_smiles = [row[col] for col in X_cols]

        smiles_all.append(cur_smiles)


    y = df[y_col]

    return smiles_all, y


def load_data(file):
    if(file == "Nature_SMILES.xlsx"):
       return load_nature_data()
    elif(file=="database_ml-new-up-up.csv"):
        return load_upup_data()

FILES = ["Nature_SMILES.xlsx", "database_ml-new-up-up.csv"]
file = FILES[1]
smiles_all, y = load_data(file)

"""# --- Load data ---
xlsx = "Nature_SMILES.xlsx"
df = pd.read_excel(xlsx, sheet_name="df")
solvent = pd.read_excel(xlsx, sheet_name='solvent')
y = df["DDG"].values
# smiles_all = df["SMILES_all"].values  # assume concatenated SMILES column

# temp fix
smiles_all = solvent["SMILES_s"].values"""


# Convert to Morgan fingerprints
X = np.array([np.hstack([smi2morgan(s, nbits=2048) for s in smiles]) for smiles in smiles_all])
X = np.vstack(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

# Initialize and run RaRF
rarf = RaRFRegressor(radius=0.4, metric="jaccard")

print("Hello2!")

out = rarf.fit_predict_shared(X_train, y_train, X_test, budget=int(y.shape[0]/10), alpha=1.2, redundancy_lambda=0.1)

print("Selected training indices:", out["selected"])
print("Average neighbors per target:", np.mean(out["neighbor_counts"]))
print("Predictions shape:", out["preds"].shape)

# Plot
rarf.plot_overlap_map(X_train, X_test, out["selected"])
print("Saved plot as overlap_map.png")


plt.clf()
plt.scatter(out['preds'], y_test)




file = open(f"./predictions/{file}", mode='w')

file.write(str(out['preds'].tolist()))
file.write("\n")
file.write(str(y.tolist()))