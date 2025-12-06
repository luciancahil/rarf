
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.ensemble import RandomForestRegressor
from scipy.spatial.distance import cdist
import os
import pickle
import lmdb

class RaRFRegressor:
    def __init__(self, radius=0.4, metric="jaccard", rf_kwargs=None, random_state=0):
        self.radius = radius
        self.metric = metric
        self.rf_kwargs = rf_kwargs or {"n_estimators": 300, "random_state": random_state}
        self.random_state = random_state

    def _pairwise_distance(self, A, B):
        print("Before dist")
        if self.metric == "jaccard":
            return cdist((A > 0).astype(bool), (B > 0).astype(bool), metric="jaccard")
        elif self.metric == "euclidean":
            return cdist(A, B, metric="euclidean")

        print("After Dist")
        raise ValueError(f"Unsupported metric: {self.metric}")

    def _pairwise_similarity(self, A, B):
        D = self._pairwise_distance(A, B)
        if self.metric == "jaccard":
            return 1.0 - D
        med = np.median(D[D > 0]) if np.any(D > 0) else 1.0
        med = med if med > 0 else 1.0
        return np.exp(-(D**2) / (2 * (med**2)))

    # fit a random forest regressor
    def _train_rf(self, X, y):
        model = RandomForestRegressor(**self.rf_kwargs)
        model.fit(X, y)
        return model

    def greedy_shared_selection(self, X_train, X_test, budget=None, alpha=1.0, redundancy_lambda=0.0,
                                costs=None, similarity_threshold=None, name = None):
        T, N = X_test.shape[0], X_train.shape[0]
        # should be 1 for everything unless specified
        costs = np.ones(N) if costs is None else costs.astype(float)


        if not os.path.isdir(name):

            env = lmdb.open('my_lmdb_database', map_size=104857600)  # Example: 100MB map_size
            print("Calculating tt similarities")

            # similar of test example i to train example j
            sim_tt = self._pairwise_similarity(X_test, X_train)

            print("Calculating tr similarities")


            # similarity of train example i to train example j
            sim_tr = self._pairwise_similarity(X_train, X_train) if redundancy_lambda > 0 else None

            print("Storing similarities")

            with env.begin(write=True) as txn:
                # Perform operations within the transaction
                txn.put("sim_tt".encode(), pickle.dumps(sim_tt))
                txn.put("sim_tr".encode(), pickle.dumps(sim_tr))
        else:
            print("Using Stored similarities")
            with env.begin(write=True) as txn:
                # Perform operations within the transaction
                sim_tt = pickle.loads(txn.get("sim_tt".encode()))
                sim_tr = pickle.loads(txn.get("sim_tr".encode()))

            

        if similarity_threshold is None:
            similarity_threshold = 1.0 - self.radius if self.metric == "jaccard" else None
        neigh_mask = sim_tt >= similarity_threshold

        selected = []

        def gain(j):
            covered = neigh_mask[:, j]
            if not np.any(covered):
                return -np.inf
            gain_val = (sim_tt[covered, j] ** alpha).sum()
            penalty = redundancy_lambda * float(np.max(sim_tr[j, selected])) if (redundancy_lambda > 0 and selected) else 0.0
            return gain_val / costs[j] - penalty

        print(budget)
        while True:
            print(len(selected))
            # if we have selected as many as our budget allows leave
            if budget and len(selected) >= budget: break
            # this is all the elements we haven't selected. If we've selected everything, leave
            remaining = [j for j in range(N) if j not in selected]
            if not remaining: break
            # find the largest gain, and add it to "selected".
            gains = np.array([gain(j) for j in remaining])
            if np.all(np.isneginf(gains)): break
            j_star = remaining[int(np.argmax(gains))]
            selected.append(j_star)

        # A T x budget matrix.
        local_sets = [sorted([j for j in selected if neigh_mask[i, j]]) for i in range(T)]
        return selected, local_sets, sim_tt, neigh_mask, similarity_threshold

    def fit_predict_shared(self, X_train, y_train, X_test, budget=None, alpha=1.0, redundancy_lambda=0.0,
                           similarity_threshold=None,
                           enrich_min_per_target: int = 1,   # if >0, add extra neighbors beyond shared to reach this #
                           enrich_from: str = "neighbors",   # "neighbors" (within tau) or "all" (top-K from all train)
                           enrich_max_extra: int = 10,        # safety cap on per-target added points
                           name: str = None
                           ):
        

        selected, local_sets, sim_tt, neigh_mask, base_tau = self.greedy_shared_selection(
            X_train, X_test, budget=budget, alpha=alpha, redundancy_lambda=redundancy_lambda,
            similarity_threshold=similarity_threshold, name=name
        )


        tau = base_tau if base_tau is not None else (1.0 - self.radius)

        preds = np.full(X_test.shape[0], np.nan)
        # number of local atoms to each.
        ks = []
        final_sets = []
        print(len(local_sets))
        for i, idx in enumerate(local_sets):
            print(i)
            # Enrich per-target set if requested.
            # That is, add more to "selected" if need be.
            idx_set = set(idx)
            if enrich_min_per_target and len(idx_set) < enrich_min_per_target:
                need = enrich_min_per_target - len(idx_set)
                if enrich_from == "neighbors":
                    candidates = np.where(sim_tt[i] >= tau)[0]
                else:
                    # choose by descending similarity
                    candidates = np.argsort(-sim_tt[i])
                # prefer candidates not already in idx_set
                extra = [j for j in candidates if j not in idx_set][:min(need, enrich_max_extra)]
                idx_set.update(extra)

            # sort the list of local atoms, and then add this set to the list
            idx_final = sorted(idx_set)
            final_sets.append(idx_final)
            ks.append(len(idx_final))
            if len(idx_final) == 0:
                continue
            """
            Traceback (most recent call last):
            File "/home/roy/Documents/rarfviz /run_rarf_visual.py", line 91, in <module>
                out = rarf.fit_predict_shared(X_train, y_train, X_test, budget=int(y.shape[0]/10), alpha=1.2, redundancy_lambda=0.1)
            File "/home/roy/Documents/rarfviz /RaRFRegressor_shared_overlap_visual.py", line 130, in fit_predict_shared
                preds[i] = float(y_loc[0])
                                ~~~~~^^^
            File "/home/roy/anaconda3/envs/rarf/lib/python3.13/site-packages/pandas/core/series.py", line 1133, in __getitem__
                return self._get_value(key)
                    ~~~~~~~~~~~~~~~^^^^^
            File "/home/roy/anaconda3/envs/rarf/lib/python3.13/site-packages/pandas/core/series.py", line 1249, in _get_value
                loc = self.index.get_loc(label)
            File "/home/roy/anaconda3/envs/rarf/lib/python3.13/site-packages/pandas/core/indexes/base.py", line 3819, in get_loc
                raise KeyError(key) from err

            What caused the above? I don't know.
            """
            print(f"i: {idx_final}")
            X_loc, y_loc = X_train[idx_final], y_train[idx_final]
            print(f"y_loc: {y_loc}")

            if X_loc.shape[0] == 1: # if we receive only 1, predict.
                preds[i] = float(y_loc[0])
            else: # if there are several, train first, and then predict
                model = self._train_rf(X_loc, y_loc)
                preds[i] = model.predict(X_test[i].reshape(1, -1))[0]
            


        return {
            "selected": selected,
            "local_sets": final_sets,  # after enrichment
            "preds": preds,
            "neighbor_counts": ks,
            "sim": sim_tt,
            "tau": tau
        }

    def plot_overlap_map(self, X_train, X_test, selected, title="RaRF Overlap Map (UMAP)"):
        # dimensionality reduction
        reducer = umap.UMAP(random_state=0, n_neighbors=15, min_dist=0.1, metric="jaccard")
        X_all = np.vstack([X_train, X_test])
        embedding = reducer.fit_transform(X_all)

        n_train = X_train.shape[0]
        # I think this is wrong?
        train_emb = embedding[:n_train]
        test_emb = embedding[n_train:]

        import matplotlib.pyplot as plt
        plt.figure(figsize=(7,6))
        # plot train embedding
        plt.scatter(train_emb[:,0], train_emb[:,1], c="lightgrey", s=20, label="Train (all)", alpha=0.6)
        # plot test embedding
        plt.scatter(test_emb[:,0], test_emb[:,1], c="red", s=40, label="Test targets", alpha=0.7)
        # plot selected
        if len(selected) > 0:
            plt.scatter(train_emb[selected,0], train_emb[selected,1], c="#027F80", s=50, label="Selected shared", edgecolor="black")
        plt.legend(frameon=False)
        plt.title(title)
        plt.tight_layout()
        plt.savefig("overlap_map.png", dpi=300)
        #plt.show()
