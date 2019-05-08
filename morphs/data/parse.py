"""Parsing functions for the stimuli names for this project"""
from __future__ import absolute_import
import morphs
import os
import numpy as np


def morph_dim(
    df, morph_dim="morph_dim", lesser_dim="lesser_dim", greater_dim="greater_dim"
):
    df[lesser_dim] = df[morph_dim].str[0]
    df[greater_dim] = df[morph_dim].str[1]


def stim_id(
    df,
    stim_id="stim_id",
    end="end",
    morph_dim="morph_dim",
    morph_pos="morph_pos",
    lesser_dim="lesser_dim",
    greater_dim="greater_dim",
):
    df[end] = df[stim_id].isin(list("abcdefghi"))
    df[morph_dim] = df[~df[end]][stim_id].str[0:2]
    df[morph_pos] = df[~df[end]][stim_id].str[2:].astype("uint8")
    morphs.data.parse.morph_dim(
        df, morph_dim=morph_dim, lesser_dim=lesser_dim, greater_dim=greater_dim
    )


def separate_endpoints(stim_id_series):
    stim_ids = stim_id_series.str.replace("[a-i]001", "")
    for motif in "abcdefgh":
        stim_ids = stim_ids.str.replace("[a-i]%s128" % (motif), motif)
    end = stim_ids.isin(list("abcdefghi"))
    return stim_ids, end


def is_inverted_dim(subj, morph_dim):
    """
    To compare behavior between subjects with different L/R associations, 
    psychometric functions are plotted as 
    P(response associated with the alphanumerically greater motif)
    This function returns whether response associated with the
    alphanumerically greater motif is R for a given subj and morph dimension
    """
    left, right = morphs.subj.TRAINING[subj].lower().split("|")
    les, gre = morph_dim
    assert (les in left) != (gre in left)
    assert (les in right) != (gre in right)
    return gre in left


def behav_data_stim_id(df, subj):
    """parses stim_id from the wav path in the behavioral data"""
    df = df[(df["response"] != "none") & (df["type_"] == "normal")]
    df["stim_id"] = df["stimulus"].str.split("/").str[-1].str[:-4]
    df = df[df["stim_id"].str.len() == 5]
    df = df[df["stim_id"].str[1:] != "_rec"]
    df["subj"] = subj
    return df


def behav_data_inverted(df):
    """
    Flips the dimensions that need inverting
    Faster than using is_inverted_dim
    """
    # Apparently groupby with categorical dtype is broken
    # See https://github.com/pandas-dev/pandas/issues/22512#issuecomment-422422573
    df["class_"] = df["class_"].astype(str)
    inverted_map = (
        df[(df["morph_pos"] == 1)]
        .groupby(["subj", "morph_dim"], observed=True)
        .agg(lambda x: x.iloc[0])["class_"]
        == "R"
    )
    df = df.join(
        inverted_map.to_frame(name="inverted"),
        on=("subj", "morph_dim"),
        how="left",
        sort=False,
    )
    df["greater_response"] = (df["response"] == "R") != (df["inverted"])
    return df


def bird_id(block_path):
    """extracts bird id from block_path"""
    name = blockpath_name(block_path)
    bird_id = name.split("__")[-1].split("_")[0]
    # misnamed one of my recording blocks when the lab started doing ephys on different species
    if bird_id == "st1107":
        bird_id = "B1107"
    return bird_id


def blockpath_name(block_path):
    """extracts block name from block_path"""
    return block_path.split(os.sep)[-1]


def recording_site(block_path):
    """parses AP, ML, and Z position from block name"""
    name = blockpath_name(block_path)
    axes = ["AP", "ML", "Z"]
    idxs = zip(axes, [(0, -2), (0, -1), (1, -1)])
    vals = {}
    for axis, (i, j) in idxs:
        vals[axis] = name.split("__")[i].split("_")[j]
        assert vals[axis][0 : len(axis)] == axis
        vals[axis] = int(vals[axis][len(axis) :])
    return tuple(vals[axis] for axis in axes)


def effective_morph(df, behavior_subj):
    """Parses df and provides cols: inverted, effective_dim, and effective_pos"""
    left, right = morphs.subj.TRAINING[behavior_subj].lower().split("|")

    df["inverted"] = df["lesser_dim"].str.match(_in_pattern(right)) & df[
        "greater_dim"
    ].str.match(_in_pattern(left))

    df["effective_dim"] = df["morph_dim"]
    df.loc[df["inverted"], "effective_dim"] = df["morph_dim"].str[::-1][df["inverted"]]

    df["effective_pos"] = df["morph_pos"]
    df.loc[df["inverted"], "effective_pos"] = 128 - (
        df[df["inverted"]]["morph_pos"] - 1
    )


def _in_pattern(string):
    return r"[" + string + r"]"


def num_shuffles(path):
    return int(path.name.split(".")[0].split("_")[-1])


def ephys_class(
    df,
    behavior_subj_label="behavior_subj",
    neural_subj_label="neural_subj",
    class_label="class",
    split_training_cond=False,
):
    """Parses df to classify neural subj into naive, trained or other training conditions"""
    for (behave_subj, subj), group in df.groupby(
        [behavior_subj_label, neural_subj_label]
    ):
        if subj == behave_subj:
            df.loc[group.index, class_label] = "self"
        elif subj not in morphs.subj.TRAINING:
            df.loc[group.index, class_label] = "naive"
        elif split_training_cond:
            if morphs.subj.TRAINING[subj] is morphs.subj.TRAINING[behave_subj]:
                df.loc[group.index, class_label] = "same training cond"
            else:
                df.loc[group.index, class_label] = "diff training cond"
        else:
            df.loc[group.index, class_label] = "trained"


def equal_spacing(pair_df):
    return pair_df.groupby("block_path").apply(
        lambda path_group: np.all(
            path_group.groupby("morph_dim").apply(
                lambda group: len(group["lesser_morph_pos"].unique()) < 10
            )
        )
    )
