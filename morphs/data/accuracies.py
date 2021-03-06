"""Functions for calculating the accuracy of individual units on the template motifs for this project"""
from __future__ import absolute_import
from __future__ import print_function
import pickle
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression

import morphs

CLUSTER_ACCURACY_CUTOFF = 0.6


def cluster_accuracy(
    cluster,
    cluster_group,
    morph_dims,
    max_num_reps,
    n_folds=10,
    n_dim=50,
    tau=0.01,
    stim_length=0.4,
):
    """Helper function to calculate the pairwise classification accuracy of template motifs"""
    accuracies = pd.DataFrame(
        index=np.arange(len(morph_dims) * n_folds),
        columns=["cluster", "morph", "i", "accuracy"],
    )
    filtered_responses = {}
    for motif, motif_group in cluster_group.groupby("stim_id"):
        trial_groups = motif_group.groupby(["recording", "stim_presentation"])
        filtered_responses[motif] = trial_groups["stim_aligned_time"].apply(
            lambda x: morphs.spikes.filtered_response(x.values, tau=tau)
        )
    t = np.linspace(0, stim_length, n_dim)
    x = {}
    for motif in "abcdefgh":
        x[motif] = np.zeros((max_num_reps, n_dim))
    for motif in filtered_responses:
        for i, fr in enumerate(filtered_responses[motif]):
            x[motif][i, :] = fr(t)

    idx = 0
    for morph in morph_dims:
        l, r = morph
        x_concat = np.append(x[l], x[r], axis=0)
        y_concat = np.append(np.zeros(max_num_reps), np.ones(max_num_reps))
        for i, (train_idx, test_idx) in enumerate(
            StratifiedKFold(y_concat, n_folds=n_folds, shuffle=True)
        ):
            model = LogisticRegression(solver="sag", warm_start=True)
            model.fit(x_concat[train_idx], y_concat[train_idx])
            y_test_hat = model.predict(x_concat[test_idx])
            accuracies.loc[idx] = [
                cluster,
                morph,
                i,
                np.mean(y_concat[test_idx] == y_test_hat),
            ]
            idx += 1
    dtypes = {"cluster": "int64", "morph": "str", "i": "int64", "accuracy": "float64"}
    for col in dtypes:
        accuracies[col] = accuracies[col].astype(dtypes[col])
    return accuracies


def gen_cluster_accuracies():
    """Generates pickle file containing the accuracy for each cluster in each recording block"""
    accuracies = {}
    with Parallel(n_jobs=morphs.parallel.N_JOBS) as parallel:
        for block_path in morphs.paths.blocks():
            print(block_path)
            spikes = morphs.load.ephys_data(block_path, collapse_endpoints=True)

            if len(spikes["recording"].unique()) >= 1:
                template_spikes = spikes[spikes["stim_id"].isin(list("abcdefgh"))]
                assert len(template_spikes) > 0
                cluster_groups = template_spikes.groupby("cluster")

                morph_dims = spikes.morph_dim.unique()
                morph_dims = morph_dims[~pd.isnull(morph_dims)]
                morph_dims.sort()

                max_num_reps = np.max(
                    [
                        len(stim_group.groupby(by=["recording", "stim_presentation"]))
                        for stim_id, stim_group in template_spikes.groupby("stim_id")
                    ]
                )

                accuracies_list = parallel(
                    delayed(cluster_accuracy)(
                        cluster, cluster_group, morph_dims, max_num_reps
                    )
                    for (cluster, cluster_group) in cluster_groups
                )

                accuracies[block_path] = pd.concat(accuracies_list)

    morphs.paths.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(morphs.paths.ACCURACIES_PKL.as_posix(), "wb") as f:
        pickle.dump(accuracies, f)


@morphs.utils.load._load(morphs.paths.ACCURACIES_PKL, gen_cluster_accuracies)
def load_cluster_accuracies():
    """Loads pickle file containing the accuracy for each cluster in each recording block"""
    accuracies = morphs.utils.load._pickle(morphs.paths.ACCURACIES_PKL)
    cluster_accuracies = {
        block_path: accuracies[block_path]
        .groupby("cluster")
        .agg(np.mean)
        .sort_values("accuracy")
        for block_path in accuracies
    }
    return accuracies, cluster_accuracies


def good_clusters(block_path, cutoff=CLUSTER_ACCURACY_CUTOFF):
    """returns a df of clusters that have accuracy > cutoff"""
    accuracies, cluster_accuracies = morphs.load.cluster_accuracies()
    block_cluster_accuracies = cluster_accuracies[block_path]
    return block_cluster_accuracies[
        block_cluster_accuracies.accuracy > cutoff
    ].index.values


def good_recs(cutoff=CLUSTER_ACCURACY_CUTOFF):
    """returns a list of blocks that have good clusters"""
    accuracies, cluster_accuracies = morphs.load.cluster_accuracies()
    return [
        block_path
        for block_path in cluster_accuracies
        if len(good_clusters(block_path, cutoff=cutoff)) > 0
    ]


if __name__ == "__main__":
    gen_cluster_accuracies()
