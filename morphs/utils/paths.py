'''Path definitions'''
from pathlib2 import Path
from glob import glob
import morphs

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
EPHYS_DIR = morphs.paths.DATA_DIR / 'ephys'
PROCESSED_DIR = DATA_DIR / "processed"
ACCURACIES_PKL = PROCESSED_DIR / "all_accuracies.pkl"


def blocks():
    if morphs.parallel.is_local():
        block_path_template = '/mnt/cube/mthielk/analysis/%s/kwik/*'
    else:
        block_path_template = str(EPHYS_DIR) + '/%s/kwik/*'

    blocks = []
    for subj in morphs.subj.EPHYS_SUBJS:
        blocks += glob(block_path_template % (subj,))
    return blocks
