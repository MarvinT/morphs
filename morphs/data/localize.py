import pandas as pd
import numpy as np
import morphs
import pickle
from six import exec_
from pathlib2 import Path
from joblib import Parallel, delayed
import ephys
import ephys.clust
from google_drive_downloader import GoogleDriveDownloader as gdd


# adapted from klustakwik
# NEVER POINT THIS AT SOMETHING YOU DONT TRUST
def _read_python(path):
    assert path.exists()
    with open(path.as_posix(), 'r') as f:
        contents = f.read()
    metadata = {}
    exec_(contents, {}, metadata)
    metadata = {k.lower(): v for (k, v) in metadata.items()}
    return metadata


def calc_loc(block_path, squared=True):
    columns = ['block_path', 'AP', 'ML', 'Z', 'cluster', 'cluster_pos', 'cluster_accuracy']
    spikes = morphs.data.load.ephys_data(block_path, collapse_endpoints=True)
    if len(spikes) == 0:
        return pd.DataFrame(columns=columns)
    waveforms, cluster_map = ephys.clust.compute_cluster_waveforms_fast(block_path, spikes)
    amps = (waveforms[:, 0, :] + waveforms[:, -1, :]) / 2 - np.min(waveforms, axis=1)
    amps /= np.max(amps, axis=0)
    if squared:
        amps = amps ** 2
    prb_files = list(Path(block_path).glob('*.prb'))
    assert len(prb_files) == 1
    prb = _read_python(prb_files[0])
    assert len(prb['channel_groups']) == 1
    for group in prb['channel_groups']:
        chans = prb['channel_groups'][group]['geometry'].keys()
        x, y = zip(*[prb['channel_groups'][group]['geometry'][k] for k in chans])
        y_hats = np.sum(amps * np.array(y).reshape((1, -1)), axis=1) / np.sum(amps, axis=1)

    d = {}
    d['block_path'] = block_path
    d['AP'], d['ML'], d['Z'] = morphs.data.parse.recording_site(block_path)
    i_cluster_map = {v: k for k, v in cluster_map.items()}
    d['cluster'] = [i_cluster_map[i] for i in range(len(cluster_map))]
    d['cluster_pos'] = y_hats
    _, cluster_accuracies = morphs.data.load.cluster_accuracies()
    d['cluster_accuracy'] = cluster_accuracies[block_path].loc[d['cluster']]['accuracy'].values
    return pd.DataFrame(data=d, columns=columns)


def generate_all_loc(parallel=False, n_jobs=morphs.parallel.N_JOBS):
    if parallel:
        all_locs = Parallel(n_jobs=n_jobs)(delayed(calc_loc)(block_path)
                                           for block_path in morphs.paths.blocks())
    else:
        all_locs = [calc_loc(block_path) for block_path in morphs.paths.blocks()]
    all_locs_df = pd.concat(all_locs, ignore_index=True)
    all_locs_df['block_path'] = all_locs_df['block_path'].astype('category')
    all_locs_df.to_pickle(morphs.paths.LOCATIONS_PKL)


def load_all_loc(prefer_download=True):
    if not morphs.paths.LOCATIONS_PKL.exists():
        if prefer_download:
            download_all_loc()
        else:
            generate_all_loc()
    with open(morphs.paths.LOCATIONS_PKL.as_posix(), 'rb') as f:
        return pickle.load(f)


def download_all_loc():
    morphs.paths.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    gdd.download_file_from_google_drive(file_id='1wLoMiKJjKPQbNF_qplqrMzHLyFCyFXn3',
                                        dest_path=morphs.paths.LOCATIONS_PKL.as_posix())
