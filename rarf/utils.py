# utils.py — flexible Excel reader + RDKit utilities for RaRF
import numpy as np
import pandas as pd

# --- RDKit bits -> numpy vector ---
def smi2morgan(smiles: str, nbits: int = 2048, radius: int = 2) -> np.ndarray:
    """
    Convert a SMILES string to a binary Morgan fingerprint (np.uint8 array of 0/1).
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit import DataStructs
    except Exception as e:
        raise ImportError("RDKit is required for smi2morgan() but was not found in this environment.") from e

    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        # return all-zeros if invalid/empty SMILES
        return np.zeros((nbits,), dtype=np.uint8)

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    arr = np.zeros((nbits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(np.uint8)

# --- Distances (Jaccard) ---
def get_distances(X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """
    Compute Jaccard distance between binary vectors in X_test and X_train.
    Returns an array of shape (n_test, n_train) with distances in [0, 1].
    """
    # Convert to boolean for safe operations
    Xt = (X_test > 0).astype(np.int32)
    Xr = (X_train > 0).astype(np.int32)

    # intersections: test x train via matrix multiply (counts of 1&1)
    inter = Xt @ Xr.T  # (n_test, n_train)

    # ones per row
    a = Xt.sum(axis=1, keepdims=True)  # (n_test, 1)
    b = Xr.sum(axis=1, keepdims=True).T  # (1, n_train)

    # union = a + b - intersection
    union = a + b - inter
    # avoid /0: if union==0 (both zero vectors), define distance=1.0
    union = union.astype(np.float32)
    inter = inter.astype(np.float32)

    with np.errstate(divide="ignore", invalid="ignore"):
        jaccard_sim = np.where(union > 0, inter / union, 0.0)
    dist = 1.0 - jaccard_sim
    # when union==0, set distance to 1.0 (maximally dissimilar)
    dist = np.where((a == 0) & (b == 0), 1.0, dist)
    return dist.astype(np.float32)

# --- Flexible Excel parsing for RaRF ---
# Accept varied sheet/column spellings. Each lookup sheet can have either:
#   (Name, SMILES)               # generic
# or (Imine, SMILES_i), (Nucleophile|nuc, SMILES_n), (Ligand, SMILES_l), (Solvent, SMILES_s)
# Column matching is case-insensitive.
_LOOKUP_SHEET_ALIASES = {
    "Imine": ["Imine", "imine"],
    "Nucleophile": ["Nucleophile", "nucleophile", "nuc"],
    "Ligand": ["Ligand", "ligand"],
    "Solvent": ["Solvent", "solvent"],
}

_LOOKUP_COL_ALIASES = {
    # for each role, a list of (name_col_candidates, smiles_col_candidates)
    "Imine":       (["Name", "name", "Imine", "imine"],        ["SMILES", "smiles", "SMILES_i", "smiles_i"]),
    "Nucleophile": (["Name", "name", "Nucleophile", "nucleophile", "nuc"], ["SMILES", "smiles", "SMILES_n", "smiles_n"]),
    "Ligand":      (["Name", "name", "Ligand", "ligand"],      ["SMILES", "smiles", "SMILES_l", "smiles_l"]),
    "Solvent":     (["Name", "name", "Solvent", "solvent"],    ["SMILES", "smiles", "SMILES_s", "smiles_s"]),
}

def _lower_map(seq):
    return {str(x).lower(): x for x in seq}

def _resolve_sheet(xl: pd.ExcelFile, aliases):
    m = _lower_map(xl.sheet_names)
    for a in aliases:
        if a.lower() in m:
            return m[a.lower()]
    # if not found directly, try fuzzy: first sheet whose lower equals any alias token
    raise KeyError(f"Sheets {aliases} not found. Available: {xl.sheet_names}")

def _resolve_col(df: pd.DataFrame, candidates):
    cmap = _lower_map(df.columns)
    for c in candidates:
        if c.lower() in cmap:
            return cmap[c.lower()]
    raise KeyError(f"None of the columns {candidates} found in {list(df.columns)}")

def _build_lookup_maps(xl: pd.ExcelFile) -> dict:
    """
    Return dict with keys 'Imine','Nucleophile','Ligand','Solvent' mapping NAME->SMILES.
    Accepts alternate column names like SMILES_i/SMILES_n/SMILES_l/SMILES_s.
    """
    maps = {}
    for role, sheet_aliases in _LOOKUP_SHEET_ALIASES.items():
        sname = _resolve_sheet(xl, sheet_aliases)
        df = xl.parse(sname)
        name_col, smi_col = _LOOKUP_COL_ALIASES[role]
        name_real = _resolve_col(df, name_col)
        smi_real  = _resolve_col(df, smi_col)
        names = df[name_real].astype(str).tolist()
        smiles = df[smi_real].astype(str).tolist()
        maps[role] = {n: s for n, s in zip(names, smiles) if isinstance(n, str)}
    return maps

def build_X_from_excel(path: str, reactions_sheet: str | None = None,
                       target_col: str = "DDG", nbits: int = 2048):
    """
    Flexible X,y builder used by RaRF baselines.
    - Reactions sheet can be 'df' or any sheet that contains target_col (case-insensitive).
    - Uses four *_SMILES columns if present (any case), otherwise maps name columns via lookup sheets.
    - Lookup sheets accept varied column names, e.g., Imine/SMILES_i, Nucleophile/SMILES_n, etc.
    Returns: X (n, 4*nbits) uint8, y (n,) float
    """
    xl = pd.ExcelFile(path)

    # pick reactions sheet
    if reactions_sheet is None:
        try:
            rxn_sheet = _resolve_sheet(xl, ["df"])
        except KeyError:
            rxn_sheet = None
            for s in xl.sheet_names:
                tmp = xl.parse(s, nrows=1)
                if any(c.lower() == target_col.lower() for c in tmp.columns):
                    rxn_sheet = s
                    break
            if rxn_sheet is None:
                raise ValueError(f"Could not find a reactions sheet with '{target_col}'.")
    else:
        rxn_sheet = _resolve_sheet(xl, [reactions_sheet])

    rxn = xl.parse(rxn_sheet)
    # resolve target column
    tgt = _resolve_col(rxn, [target_col, target_col.lower(), target_col.upper()])

    # Option A: four *_SMILES columns present
    smiles_spec = [
        ["Imine_SMILES", "imine_smiles"],
        ["Nucleophile_SMILES", "nucleophile_smiles", "nuc_smiles"],
        ["Ligand_SMILES", "ligand_smiles"],
        ["Solvent_SMILES", "solvent_smiles"],
    ]
    smiles_cols = []
    ok = True
    for cands in smiles_spec:
        try:
            smiles_cols.append(_resolve_col(rxn, cands))
        except KeyError:
            ok = False
            smiles_cols.append(None)

    if ok:
        parts = [rxn[c] for c in smiles_cols]
        y = rxn[tgt].to_numpy(dtype=float)
        X = np.zeros((len(rxn), nbits * 4), dtype=np.uint8)
        for i in range(len(rxn)):
            bits = [smi2morgan(str(parts[j].iloc[i]), nbits=nbits) for j in range(4)]
            X[i] = np.concatenate(bits, axis=0)
        return X, y

    # Option B: name columns + flexible lookup sheets
    name_cols = {
        "Imine":       ["Imine", "imine"],
        "Nucleophile": ["Nucleophile", "nucleophile", "nuc"],
        "Ligand":      ["Ligand", "ligand"],
        "Solvent":     ["Solvent", "solvent"],
    }
    rxn_name_real = {role: _resolve_col(rxn, cands) for role, cands in name_cols.items()}
    lut = _build_lookup_maps(xl)

    y = rxn[tgt].to_numpy(dtype=float)
    X = np.zeros((len(rxn), nbits * 4), dtype=np.uint8)

    missing = []
    for i, row in rxn.iterrows():
        names = {
            "Imine":       str(row[rxn_name_real["Imine"]]),
            "Nucleophile": str(row[rxn_name_real["Nucleophile"]]),
            "Ligand":      str(row[rxn_name_real["Ligand"]]),
            "Solvent":     str(row[rxn_name_real["Solvent"]]),
        }
        smis = [
            lut["Imine"].get(names["Imine"], ""),
            lut["Nucleophile"].get(names["Nucleophile"], ""),
            lut["Ligand"].get(names["Ligand"], ""),
            lut["Solvent"].get(names["Solvent"], ""),
        ]
        for role, nm, sm in zip(["Imine","Nucleophile","Ligand","Solvent"], names.values(), smis):
            if not sm:
                missing.append((i, role, nm))
        bits = [smi2morgan(s, nbits=nbits) for s in smis]
        X[i] = np.concatenate(bits, axis=0)

    if missing:
        # Warn but proceed
        summary = {}
        for _, role, nm in missing:
            summary.setdefault(role, set()).add(nm)
        msg = "WARNING: Missing SMILES in lookup tables:\n" + "\n".join(
            f"  {role}: {', '.join(list(sorted(v))[:10])}{' ...' if len(v)>10 else ''}" for role, v in summary.items()
        )
        print(msg)

    return X, y
