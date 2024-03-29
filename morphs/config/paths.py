"""Path definitions"""
from __future__ import absolute_import
from pathlib2 import Path
from glob import glob
import morphs

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"

STIM_DIR = DATA_DIR / "stimuli"
SPECT_PKL = STIM_DIR / "spectrograms.pkl"
WAV_ZIP = STIM_DIR / "wavs.zip"
WAV_DIR = STIM_DIR / "wavs"

BEHAVE_DIR = DATA_DIR / "behavior"
BEHAVE_PKL = BEHAVE_DIR / "behave.pkl"

EPHYS_DIR = DATA_DIR / "ephys"

PROCESSED_DIR = DATA_DIR / "processed"
WAVEFORMS_PKL = PROCESSED_DIR / "waveforms.pkl"
SINGLE_UNIT_TEMPLATES_PKL = PROCESSED_DIR / "single_unit_templates.pkl"
PSYCHOMETRIC_PKL = PROCESSED_DIR / "psychometrics.pkl"
LOCATIONS_PKL = PROCESSED_DIR / "unit_locations.pkl"
ACCURACIES_PKL = PROCESSED_DIR / "all_accuracies.pkl"
POP_PAIR_PKL = PROCESSED_DIR / "pop_pair_neural_dist.pkl"
DERIVATIVE_PKL = PROCESSED_DIR / "derivative.pkl"

NEUROMETRIC_NULL_DIR = PROCESSED_DIR / "neurometric_null"

KS_DF_DIR = PROCESSED_DIR / "KS_df"

EPHYS_MEMO_DIR = PROCESSED_DIR / "ephys_memo"

REPORTS_DIR = PROJECT_DIR / "reports"

FIGURES_DIR = REPORTS_DIR / "figures"


def blocks():
    """
    Returns a list of different block paths that correspond to each
    electrophysiological recording Ive done. Understands if this is
    being run with local access to data.
    """
    if morphs.parallel.is_local():
        block_path_template = "/mnt/cube/mthielk/analysis/%s/kwik/*"
    else:
        block_path_template = EPHYS_DIR.as_posix() + "/%s/kwik/*"

    blocks = []
    for subj in morphs.subj.EPHYS_SUBJS:
        blocks += glob(block_path_template % (subj,))
    return blocks


def behave_data_folder():
    if morphs.parallel.is_local():
        return "/mnt/cube/RawData/Zog/"
    else:
        return BEHAVE_DIR.as_posix()


def num_shuffle_dir(num_shuffles):
    return NEUROMETRIC_NULL_DIR / str(num_shuffles)


def num_shuffle_pkl(num_shuffles):
    return NEUROMETRIC_NULL_DIR / (
        "neurometric_null_dist_" + str(num_shuffles) + ".pkl"
    )


def psych_shuffle_pkl(num_shuffles):
    return PROCESSED_DIR / (
        "psychometrics_shuffled_" + str(num_shuffles) + ".pkl"
    )


def ks_df_pkl(num_shuffles):
    return KS_DF_DIR / ("ks_df_" + str(num_shuffles) + ".pkl")


def ephys_pkl(block_path, collapse_endpoints):
    fname = morphs.data.parse.blockpath_name(block_path)
    if collapse_endpoints:
        fname += "_collapsed"
    fname += ".pkl"
    return EPHYS_MEMO_DIR / fname
