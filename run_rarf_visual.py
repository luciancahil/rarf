
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from RaRFRegressor_shared_overlap_visual import RaRFRegressor
from utils import smi2morgan

# --- Load data ---
df = pd.read_excel("Nature_SMILES.xlsx", sheet_name="df")
y = df["DDG"].values
smiles_all = df["SMILES_all"].values  # assume concatenated SMILES column

# Convert to Morgan fingerprints
X = np.array([smi2morgan(s, nbits=2048) for s in smiles_all])
X = np.vstack(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

# Initialize and run RaRF
rarf = RaRFRegressor(radius=0.4, metric="jaccard")
out = rarf.fit_predict_shared(X_train, y_train, X_test, budget=30, alpha=1.2, redundancy_lambda=0.1)

print("Selected training indices:", out["selected"])
print("Average neighbors per target:", np.mean(out["neighbor_counts"]))
print("Predictions shape:", out["preds"].shape)

# Plot
rarf.plot_overlap_map(X_train, X_test, out["selected"])
print("Saved plot as overlap_map.png")
