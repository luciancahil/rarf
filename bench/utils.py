# utils.py — flexible Excel loader + RDKit + (optional) distances
import numpy as np
import pandas as pd

def smi2morgan(smiles: str, nbits: int = 2048, radius: int = 2) -> np.ndarray:
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit import DataStructs
    except Exception as e:
        raise ImportError("RDKit not found. Install RDKit to use smi2morgan().") from e
    smi = smiles or ""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.zeros((nbits,), dtype=np.uint8)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    arr = np.zeros((nbits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(np.uint8)

def get_distances(X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    Xt = (X_test > 0).astype(np.int32)
    Xr = (X_train > 0).astype(np.int32)
    inter = Xt @ Xr.T
    a = Xt.sum(axis=1, keepdims=True)
    b = Xr.sum(axis=1, keepdims=True).T
    union = (a + b - inter).astype(np.float32)
    inter = inter.astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = np.where(union > 0, inter / union, 0.0)
    dist = 1.0 - sim
    dist = np.where((a == 0) & (b == 0), 1.0, dist)
    return dist.astype(np.float32)

_SHEET_ALIASES = {
    "reactions": ["df", "reactions", "Reactions", "sheet1", "Sheet1"],
    "Imine": ["Imine", "imine", "imines", "Imines"],
    "Nucleophile": ["Nucleophile", "nucleophile", "nuc", "Nuc", "nucleophiles", "Nucleophiles"],
    "Ligand": ["Ligand", "ligand", "ligands", "Ligands"],
    "Solvent": ["Solvent", "solvent", "solvents", "Solvents"],
}

_ROLE_COLS = {
    "Imine": ["Imine", "imine"],
    "Nucleophile": ["Nucleophile", "nucleophile", "nuc", "Nuc"],
    "Ligand": ["Ligand", "ligand"],
    "Solvent": ["Solvent", "solvent"],
}
_LOOKUP_NAME_COLS = ["Name", "Label", "Compound", "name", "label", "compound", "Imine", "Nucleophile", "Ligand", "Solvent", "imine", "nucleophile", "ligand", "solvent"]
_LOOKUP_SMILES_COLS = ["SMILES", "Smiles", "smiles", "SMILES_i", "SMILES_n", "SMILES_l", "SMILES_s",
                       "smiles_i", "smiles_n", "smiles_l", "smiles_s"]

_REACTION_DIRECT_SMILES = [
    ("Imine_SMILES", ["Imine_SMILES", "imine_smiles"]),
    ("Nucleophile_SMILES", ["Nucleophile_SMILES", "nucleophile_smiles", "nuc_smiles"]),
    ("Ligand_SMILES", ["Ligand_SMILES", "ligand_smiles"]),
    ("Solvent_SMILES", ["Solvent_SMILES", "solvent_smiles"]),
]

def _lower_map(seq):
    return {str(x).strip().lower(): x for x in seq}

def _resolve_sheet(xl, aliases) -> str:
    m = _lower_map(xl.sheet_names)
    for a in aliases:
        if a.lower() in m:
            return m[a.lower()]
    raise KeyError(f"None of {aliases} found; available sheets: {xl.sheet_names}")

def _resolve_sheet_auto(xl) -> str:
    try:
        return _resolve_sheet(xl, _SHEET_ALIASES["reactions"])
    except KeyError:
        return xl.sheet_names[0]

def _resolve_col(df, candidates) -> str:
    cmap = _lower_map(df.columns)
    for c in candidates:
        if c.lower() in cmap:
            return cmap[c.lower()]
    for col in df.columns:
        for c in candidates:
            if c.lower() in str(col).lower():
                return col
    raise KeyError(f"None of {candidates} found in columns: {list(df.columns)}")

def _load_lookup_map(xl, sheet_aliases) -> dict:
    sname = _resolve_sheet(xl, sheet_aliases)
    df = xl.parse(sname)
    name_col = _resolve_col(df, _LOOKUP_NAME_COLS)
    smi_col = _resolve_col(df, _LOOKUP_SMILES_COLS)
    df = df[[name_col, smi_col]].dropna().copy()
    df[name_col] = df[name_col].astype(str).str.strip()
    df[smi_col] = df[smi_col].astype(str).str.strip()
    mp = {}
    for _, r in df.iterrows():
        mp[r[name_col]] = r[smi_col]
    return mp

def build_X_from_excel(xlsx_path: str, reactions_sheet: str | None = None,
                       target_col: str = "DDG", nbits: int = 2048):
    xl = pd.ExcelFile(xlsx_path)
    if reactions_sheet:
        try:
            rxn_sheet = _resolve_sheet(xl, [reactions_sheet])
        except KeyError:
            rxn_sheet = _resolve_sheet_auto(xl)
    else:
        rxn_sheet = _resolve_sheet_auto(xl)

    rxn = xl.parse(rxn_sheet)
    target_real = _resolve_col(rxn, [target_col, target_col.lower(), target_col.upper()])

    # Option A: direct *_SMILES columns
    smiles_cols = []
    direct_ok = True
    for _, cand_list in _REACTION_DIRECT_SMILES:
        try:
            smiles_cols.append(_resolve_col(rxn, cand_list))
        except KeyError:
            direct_ok = False
            smiles_cols.append(None)

    if direct_ok:
        y = rxn[target_real].to_numpy(dtype=float)
        X = np.zeros((len(rxn), nbits * 4), dtype=np.uint8)
        for i in range(len(rxn)):
            bits = [smi2morgan(str(rxn[sm_col].iloc[i]), nbits=nbits) for sm_col in smiles_cols]
            X[i] = np.concatenate(bits, axis=0)
        return X, y

    # Option B: names + lookups
    role_cols = {role: _resolve_col(rxn, cols) for role, cols in _ROLE_COLS.items()}
    L_imi = _load_lookup_map(xl, _SHEET_ALIASES["Imine"])
    L_nuc = _load_lookup_map(xl, _SHEET_ALIASES["Nucleophile"])
    L_lig = _load_lookup_map(xl, _SHEET_ALIASES["Ligand"])
    L_sol = _load_lookup_map(xl, _SHEET_ALIASES["Solvent"])

    names = {
        "Imine": rxn[role_cols["Imine"]].astype(str).str.strip().tolist(),
        "Nucleophile": rxn[role_cols["Nucleophile"]].astype(str).str.strip().tolist(),
        "Ligand": rxn[role_cols["Ligand"]].astype(str).str.strip().tolist(),
        "Solvent": rxn[role_cols["Solvent"]].astype(str).str.strip().tolist(),
    }
    lookups = {"Imine": L_imi, "Nucleophile": L_nuc, "Ligand": L_lig, "Solvent": L_sol}

    y = rxn[target_real].to_numpy(dtype=float)
    X = np.zeros((len(rxn), nbits * 4), dtype=np.uint8)

    missing = []
    for i in range(len(rxn)):
        smis = []
        for role in ("Imine", "Nucleophile", "Ligand", "Solvent"):
            nm = names[role][i]
            sm = lookups[role].get(nm, "")
            if not sm:
                missing.append((role, nm))
            smis.append(sm)
        bits = [smi2morgan(s, nbits=nbits) for s in smis]
        X[i] = np.concatenate(bits, axis=0)

    if missing:
        by_role = {}
        for role, nm in missing:
            by_role.setdefault(role, set()).add(nm)
        msg = "WARNING: Missing SMILES in lookups:\n" + "\n".join(
            f"  {role}: {', '.join(sorted(list(v))[:10])}{' ...' if len(v)>10 else ''}"
            for role, v in by_role.items()
        )
        print(msg)

    return X, y
