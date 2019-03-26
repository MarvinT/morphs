from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import morphs
import pandas as pd
import itertools
from joblib import Parallel, delayed
import click
from scipy.spatial.distance import euclidean, correlation, cosine, pdist
import datetime
import resource


def calculate_pair_df(X, labels, reduced=False, del_columns=True):
    metrics = [
        (euclidean, "euclidean"),
        (correlation, "correlation"),
        (cosine, "cosine"),
    ]
    spects = morphs.load.morph_spectrograms()
    spect_reps = {
        "%s%s%03d" % (l, g, i): spects[l][g][i][:, :, 0].astype(float)
        for l in spects
        for g in spects[l]
        for i in spects[l][g]
    }

    label_df = pd.DataFrame(data={"stim_id": labels})
    morphs.data.parse.stim_id(label_df)

    if reduced:
        X_red = morphs.data.neurometric.logistic_dim_reduction(X, labels)

    df_list = []
    for morph_dim, group in label_df.groupby("morph_dim"):
        morph_pair_df = pd.DataFrame.from_records(
            [
                (morph_dim, i1, i2)
                for i1, i2 in itertools.combinations(group.index.values, 2)
            ],
            columns=("morph_dim", "lesser_index", "greater_index"),
        )
        for dist, dist_name in metrics:
            morph_pair_df["neural_%s_dist" % (dist_name)] = pdist(
                X[group.index.values, :], metric=dist_name
            )
            if reduced:
                morph_pair_df["red_neural_%s_dist" % (dist_name)] = pdist(
                    X_red[group.index.values, :], metric=dist_name
                )
        df_list.append(morph_pair_df)
    pair_df = pd.concat(df_list, ignore_index=True)
    morphs.data.parse.morph_dim(pair_df)

    for index in ["greater", "lesser"]:
        pair_df[index + "_morph_pos"] = label_df.loc[
            pair_df[index + "_index"].values, "morph_pos"
        ].values

    pair_df["morph_dist"] = pair_df["greater_morph_pos"] - pair_df["lesser_morph_pos"]

    for (morph_dim, lmp, gmp), group in pair_df.groupby(
        ["morph_dim", "lesser_morph_pos", "greater_morph_pos"]
    ):
        lsr = spect_reps["%s%03d" % (morph_dim, lmp)].reshape(-1)
        gsr = spect_reps["%s%03d" % (morph_dim, gmp)].reshape(-1)
        for dist, dist_name in metrics:
            pair_df.loc[group.index, "spect_%s_dist" % (dist_name)] = dist(lsr, gsr)

    if del_columns:
        for col in ["lesser_index", "greater_index", "lesser_dim", "greater_dim"]:
            del pair_df[col]
    pair_df["morph_dim"] = pair_df["morph_dim"].astype("category")

    return pair_df


def calculate_pop_pair_df(block_path):
    good_clusters = morphs.data.accuracies.good_clusters(block_path)
    spikes = morphs.load.ephys_data(block_path, good_clusters=good_clusters)
    X, labels = morphs.spikes.create_neural_rep(spikes, max_id_len=5)
    pair_df = calculate_pair_df(X, labels, reduced=True)
    pair_df["block_path"] = block_path
    pair_df["block_path"] = pair_df["block_path"].astype("category")
    pair_df["block_path"].cat.set_categories(morphs.data.accuracies.good_recs())
    return pair_df


def gen_pop_pair_df(parallel=False, n_jobs=morphs.parallel.N_JOBS):
    if parallel and n_jobs > 1:
        all_pairs = Parallel(n_jobs=n_jobs)(
            delayed(calculate_pop_pair_df)(block)
            for block in morphs.data.accuracies.good_recs()
        )
    else:
        all_pairs = [
            calculate_pop_pair_df(block) for block in morphs.data.accuracies.good_recs()
        ]
    all_pairs_df = pd.concat(all_pairs, ignore_index=True)
    all_pairs_df["block_path"] = all_pairs_df["block_path"].astype("category")

    morphs.paths.POP_PAIR_PKL.parent.mkdir(parents=True, exist_ok=True)
    all_pairs_df.to_pickle(morphs.paths.POP_PAIR_PKL)


@morphs.utils.load._load(morphs.paths.POP_PAIR_PKL, gen_pop_pair_df)
def load_pop_pair_df():
    return pd.read_pickle(morphs.paths.POP_PAIR_PKL.as_posix())


@click.command()
@click.option(
    "--parallel",
    "-p",
    is_flag=True,
    help="whether to parallelize each block to its own process",
)
@click.option(
    "--num_jobs", default=morphs.parallel.N_JOBS, help="number of parallel cores to use"
)
def _main(parallel, num_jobs):
    tstart = datetime.datetime.now()
    gen_pop_pair_df(parallel=parallel, n_jobs=num_jobs)
    print(
        "peak memory usage: %f GB"
        % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)
    )
    print("time: %s" % (datetime.datetime.now() - tstart))


if __name__ == "__main__":
    _main()
