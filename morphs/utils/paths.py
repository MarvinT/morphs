'''Path definitions'''
from pathlib import Path
from glob import glob
import morphs

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
ACCURACIES_PKL = PROCESSED_DIR / "all_accuracies.pkl"

BLOCK_PATH_TEMPLATE = '/mnt/cube/mthielk/analysis/%s/kwik/*'

BLOCKS = []
for subj in morphs.subj.EPHYS_SUBJS:
    BLOCKS += glob(BLOCK_PATH_TEMPLATE % (subj,))
