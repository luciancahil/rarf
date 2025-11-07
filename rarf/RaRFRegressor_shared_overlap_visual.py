
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

class RaRFRegressor:
    """
    RaRF-Shared:
      - compute Jaccard similarities between test and train bit vectors
      - pick a shared subset of training indices with a greedy coverage score
      - per target: build a local RF on selected∩neighbors, then enrich if below a floor
    """
    def __init__(self, radius=0.4, metric="jaccard", n_estimators=200, random_state=42):
        assert metric in ("jaccard",)
        self.radius = radius
        self.metric = metric
        self.n_estimators = n_estimators
        self.random_state = random_state

    def _similarity(self, X_test, X_train):
        # pairwise_distances with metric="jaccard" expects boolean/0-1
        Xt = (X_test > 0).astype(np.uint8)
        Xr = (X_train > 0).astype(np.uint8)
        D = pairwise_distances(Xt, Xr, metric="jaccard", n_jobs=1)
        S = 1.0 - D
        return S

    def _greedy_select(self, S, budget=30, alpha=1.2, redundancy_lambda=0.1, tau=None):
        """
        S: (n_test, n_train) similarity; tau converts to in-radius (S>=tau)
        score for a candidate train j = (#targets i with S[i,j]>=tau)^alpha - redundancy*overlap
        """
        n_test, n_train = S.shape
        tau = 1.0 - self.radius if tau is None else tau
        in_rad = (S >= tau).astype(np.uint8)

        selected = []
        covered_count = np.zeros(n_test, dtype=np.int32)

        for _ in range(min(budget, n_train)):
            # marginal gain for each j not yet selected
            best_j = None
            best_gain = -1e9
            for j in range(n_train):
                if j in selected: 
                    continue
                # targets this j would cover
                col = in_rad[:, j]
                gain_coverage = col.sum() ** alpha
                # redundancy penalty = overlap with what we already covered
                red = (covered_count > 0) & (col > 0)
                redundancy = red.sum()
                gain = gain_coverage - redundancy_lambda * redundancy
                if gain > best_gain:
                    best_gain = gain
                    best_j = j
            if best_j is None:
                break
            selected.append(best_j)
            covered_count += in_rad[:, best_j]

        return np.array(selected, dtype=int), tau, in_rad

    def fit_predict_shared(self, X_train, y_train, X_test,
                           budget=30, alpha=1.2, redundancy_lambda=0.1,
                           enrich_min_per_target=5, enrich_from="neighbors", enrich_max_extra=10):
        S = self._similarity(X_test, X_train)
        selected, tau, in_rad = self._greedy_select(S, budget, alpha, redundancy_lambda)
        n_test = X_test.shape[0]
        preds = np.full(n_test, np.nan)
        neighbor_counts = np.zeros(n_test, dtype=int)
        local_sets = []

        for i in range(n_test):
            nbr_all = np.where(S[i] >= tau)[0].tolist()
            nbr_sel = [j for j in nbr_all if j in selected]
            train_idx = list(nbr_sel)

            # enrichment if below floor
            if len(train_idx) < enrich_min_per_target and enrich_from == "neighbors":
                # take the remaining neighbors sorted by similarity (desc)
                rest = [j for j in nbr_all if j not in train_idx]
                rest_sorted = sorted(rest, key=lambda j: S[i, j], reverse=True)
                need = min(enrich_min_per_target - len(train_idx), enrich_max_extra, len(rest_sorted))
                train_idx += rest_sorted[:need]

            neighbor_counts[i] = len(train_idx)
            local_sets.append(train_idx)

            if len(train_idx) > 0:
                rf = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state, n_jobs=-1)
                rf.fit(X_train[train_idx], y_train[train_idx])
                preds[i] = rf.predict(X_test[i:i+1])[0]

        return {
            "preds": preds,
            "sim": S,
            "selected": selected.tolist(),
            "tau": tau,
            "neighbor_counts": neighbor_counts.tolist(),
            "local_sets": local_sets,
        }

    def plot_overlap_map(self, X_train, X_test, selected_idx, out_path="overlap_map.png"):
        # Use UMAP if available, else PCA fallback to avoid import errors
        try:
            import umap
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="jaccard", random_state=42)
            Emb = reducer.fit_transform(np.vstack([(X_train>0).astype(np.uint8), (X_test>0).astype(np.uint8)]))
            n_train = X_train.shape[0]
            E_train, E_test = Emb[:n_train], Emb[n_train:]
        except Exception:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=42)
            Emb = pca.fit_transform(np.vstack([(X_train>0).astype(np.uint8), (X_test>0).astype(np.uint8)]))
            n_train = X_train.shape[0]
            E_train, E_test = Emb[:n_train], Emb[n_train:]

        plt.figure(figsize=(7,6))
        plt.scatter(E_train[:,0], E_train[:,1], s=20, c="#d9d9d9", label="Train (all)")
        plt.scatter(E_test[:,0], E_test[:,1], s=20, c="tomato", label="Test targets")
        if selected_idx is not None and len(selected_idx)>0:
            plt.scatter(E_train[selected_idx,0], E_train[selected_idx,1],
                        s=50, facecolors="none", edgecolors="#008b8b", linewidths=1.5, label="Selected shared")
        plt.legend()
        plt.title("RaRF Overlap Map")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
