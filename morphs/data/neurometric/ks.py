from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import pandas as pd
import scipy as sp
import morphs
from joblib import Parallel, delayed


def gen_block_ks_df(block_path, block_group):
    print(block_path)
    shuffled_df, unshuffle_df = shuffle_ks_df(block_group)
    ks_df = merge_ks_dfs(shuffled_df, unshuffle_df)
    ks_df["neuro_subj"] = morphs.data.parse.bird_id(block_path)
    ks_df["block_path"] = block_path
    return ks_df


def shuffle_ks_df(samples_df):
    shuffled_grouped = samples_df.groupby("shuffled")
    shuffled = shuffled_grouped.get_group(True)["errors"].values
    # fit = shuffled_grouped.get_group(False)['errors'].values

    for was_shuffled, shuffle_group in shuffled_grouped:
        grouped = shuffle_group.groupby(["shuffle_index", "subj"])
        temp_results = np.zeros(len(grouped))
        temp_p_results = np.zeros(len(grouped))
        temp_subj_results = ["" for i in range(len(grouped))]
        for i, ((shuffle_index, subj), group) in enumerate(grouped):
            temp_results[i], temp_p_results[i] = sp.stats.mstats.ks_twosamp(
                group["errors"].values, shuffled, alternative="greater"
            )
            temp_subj_results[i] = subj

        df = pd.DataFrame(columns=["subj", "ks_stat", "ks_p"])
        df["subj"] = temp_subj_results
        df["ks_stat"] = temp_results
        df["ks_p"] = temp_p_results

        if was_shuffled:
            shuffled_df = df
        else:
            unshuffle_df = df

    return shuffled_df, unshuffle_df


def merge_ks_dfs(shuffled_df, unshuffle_df, reset_index=True):
    temp_df = shuffled_df.merge(
        unshuffle_df, on=("subj"), suffixes=("_shuffled", "_unshuffled")
    )
    temp_df["p"] = temp_df["ks_stat_shuffled"] > temp_df["ks_stat_unshuffled"]
    ks_df = temp_df.groupby("subj").apply(np.mean)
    if reset_index:
        ks_df = ks_df.reset_index()
    del ks_df["ks_stat_shuffled"]
    return ks_df


def generate_ks_df(num_shuffles, parallel=True, n_jobs=morphs.parallel.N_JOBS):
    all_samples_df = morphs.load.neurometric_null_all(num_shuffles)

    if parallel and n_jobs > 1:
        all_ks = Parallel(n_jobs=n_jobs)(
            delayed(gen_block_ks_df)(block, block_group)
            for block, block_group in all_samples_df.groupby("block")
        )
    else:
        all_ks = [
            gen_block_ks_df(block, block_group)
            for block, block_group in all_samples_df.groupby("block")
        ]

    all_ks_df = pd.concat(all_ks, ignore_index=True)
    # misnamed one of my recording blocks when the lab started doing ephys on different species
    all_ks_df.loc[all_ks_df["neuro_subj"] == "st1107", "neuro_subj"] = "B1107"

    for (behave_subj, subj), group in all_ks_df.groupby(["subj", "neuro_subj"]):
        if subj == behave_subj:
            all_ks_df.loc[group.index, "class"] = "self"
        elif subj not in morphs.subj.TRAINING:
            all_ks_df.loc[group.index, "class"] = "naive"
        elif morphs.subj.TRAINING[subj] is morphs.subj.TRAINING[behave_subj]:
            all_ks_df.loc[group.index, "class"] = "same training cond"
        else:
            all_ks_df.loc[group.index, "class"] = "diff training cond"

    morphs.paths.ks_df_pkl(num_shuffles).parent.mkdir(parents=True, exist_ok=True)
    all_ks_df.to_pickle(morphs.paths.ks_df_pkl(num_shuffles))


@morphs.utils.load._load(morphs.paths.ks_df_pkl, generate_ks_df)
def load_ks_df(num_shuffles):
    return pd.read_pickle(morphs.paths.ks_df_pkl(num_shuffles).as_posix())


if __name__ == "__main__":
    for path in morphs.paths.NEUROMETRIC_NULL_DIR.glob("*.pkl"):
        num_shuffles = morphs.data.parse.num_shuffles(path)
        generate_ks_df(num_shuffles)
