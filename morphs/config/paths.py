'''Path definitions'''
from __future__ import absolute_import
from pathlib2 import Path
from glob import glob
import morphs

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
EPHYS_DIR = DATA_DIR / 'ephys'
BEHAVE_DIR = DATA_DIR / 'behavior'
BEHAVE_PKL = BEHAVE_DIR / 'behave.pkl'
PROCESSED_DIR = DATA_DIR / "processed"
ACCURACIES_PKL = PROCESSED_DIR / "all_accuracies.pkl"
PSYCHOMETRIC_PKL = PROCESSED_DIR / "psychometrics.pkl"
NEUROMETRIC_NULL_DIR = PROCESSED_DIR / "neurometric_null"

REPORTS_DIR = PROJECT_DIR / 'reports'
FIGURES_DIR = REPORTS_DIR / 'figures'


def blocks():
    '''
    Returns a list of different block paths that correspond to each
    electrophysiological recording Ive done. Understands if this is
    being run with local access to data.
    '''
    if morphs.parallel.is_local():
        block_path_template = '/mnt/cube/mthielk/analysis/%s/kwik/*'
    else:
        block_path_template = EPHYS_DIR.as_posix() + '/%s/kwik/*'

    blocks = []
    for subj in morphs.subj.EPHYS_SUBJS:
        blocks += glob(block_path_template % (subj,))
    return blocks


def behave_data_folder():
    if morphs.parallel.is_local():
        return '/mnt/cube/RawData/Zog/'
    else:
        return BEHAVE_DIR.as_posix()


def num_shuffle_dir(num_shuffles):
    return NEUROMETRIC_NULL_DIR / str(num_shuffles)


def num_shuffle_pkl(num_shuffles):
    return NEUROMETRIC_NULL_DIR / ('neurometric_null_dist_' + str(num_shuffles) + '.pkl')
