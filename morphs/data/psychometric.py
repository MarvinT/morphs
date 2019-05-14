from __future__ import absolute_import
from __future__ import print_function
import pickle
import morphs
import numpy as np
import scipy as sp
import pandas as pd
import click


def calculate_psychometric_params(behavior_df):
    """Fits psychometric params for each dim x each bird"""
    psychometric_params = {}
    for subj, subj_group in behavior_df.groupby("subj"):
        psychometric_params[subj] = {}
        for dim, dim_group in subj_group.groupby("morph_dim"):
            x = dim_group["morph_pos"].astype(float).values
            y = dim_group["greater_response"].astype(float).values
            psychometric_params[subj][dim] = morphs.logistic.fit_4pl(x, y, verbose=True)
    return psychometric_params


def generate_psychometric_params():
    """
    Loads behavioral data, parses, calculates psychometric params,
    then drops them into a pkl file
    """
    behavior_df = morphs.load.behavior_df()
    morphs.data.parse.stim_id(behavior_df)
    behavior_df = morphs.data.parse.behav_data_inverted(behavior_df)
    psychometric_params = calculate_psychometric_params(
        behavior_df[behavior_df["type_"] == "normal"]
    )
    morphs.paths.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(morphs.paths.PSYCHOMETRIC_PKL.as_posix(), "wb") as f:
        pickle.dump(psychometric_params, f)


@morphs.utils.load._load(morphs.paths.PSYCHOMETRIC_PKL, generate_psychometric_params)
def load_psychometric_params():
    """loads pickle file containing the fit psychometric parameters for each bird"""
    return morphs.utils.load._pickle(morphs.paths.PSYCHOMETRIC_PKL)


def load_psychometric_df():
    psychometric_params = morphs.load.psychometric_params()
    nrows = len(psychometric_params) * 16
    df = pd.DataFrame(columns=['subj', 'training', 'dim', 'A',
                               'K', 'B', 'M'], index=np.arange(0, nrows))
    ind = 0
    for subj in psychometric_params:
        for dim in psychometric_params[subj]:
            df.loc[ind] = [subj, morphs.config.subj.TRAINING[
                subj], dim] + list(psychometric_params[subj][dim])
            ind += 1

    df = df[:ind]
    df = df.infer_objects()
    return df


def load_psychometric_param_pdists():
    psychometric_df = load_psychometric_df()
    distributions = {}
    for i, param in enumerate(["A", "K", "B", "M"]):
        distributions[param] = {}
        distributions[param]["null_dist"] = sp.spatial.distance.pdist(
            psychometric_df[param].values.reshape(-1, 1)
        )
        for split in ["subj", "dim"]:
            distributions[param][split] = np.concatenate(
                [
                    sp.spatial.distance.pdist(group[param].values.reshape(-1, 1))
                    for _, group in psychometric_df.groupby(split)
                ]
            )
    return distributions


def gen_psychometric_param_shuffled_pdists(num_shuffles=1024):
    psychometric_df = load_psychometric_df()
    shuffled_df = load_psychometric_df()
    distributions = load_psychometric_param_pdists()
    null_kstats = {}
    for param in ["A", "K", "B", "M"]:
        null_kstats[param] = {}
        for split in ["subj", "dim"]:
            null_kstats[param][split] = np.zeros(num_shuffles)
            for i in range(num_shuffles):
                shuffled_df[split] = np.random.permutation(psychometric_df[split])
                sample = np.concatenate(
                    [
                        sp.spatial.distance.pdist(group[param].values.reshape(-1, 1))
                        for _, group in shuffled_df.groupby(split)
                    ]
                )
                null_kstats[param][split][i] = sp.stats.mstats.ks_twosamp(
                    sample, distributions[param]["null_dist"], alternative="greater"
                )[0]
    with open(morphs.paths.psych_shuffle_pkl(num_shuffles).as_posix(), "wb") as f:
        pickle.dump(null_kstats, f)


@morphs.utils.load._load(
    morphs.paths.psych_shuffle_pkl, gen_psychometric_param_shuffled_pdists
)
def load_psychometric_param_shuffled_pdists(num_shuffles):
    return morphs.utils.load._pickle(morphs.paths.psych_shuffle_pkl(num_shuffles))


@click.command()
@click.option(
    "--num_shuffles", default=1024, help="number of times to shuffle null distribution"
)
def _main(num_shuffles):
    psychometric_params = morphs.load.psychometric_params()
    null_kstats = load_psychometric_param_shuffled_pdists(num_shuffles)


if __name__ == "__main__":
    _main()
