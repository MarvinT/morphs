from __future__ import absolute_import
import morphs
import pickle
from joblib import Parallel, delayed
import ephys
import ephys.clust


def calc_waveforms(block_path):
    spikes = morphs.load.ephys_data(block_path)
    if len(spikes) == 0:
        return block_path, (None, None)
    waveforms, cluster_map = ephys.clust.compute_cluster_waveforms_fast(block_path, spikes)
    return block_path, (waveforms, cluster_map)


def generate_waveforms(parallel=False, n_jobs=morphs.parallel.N_JOBS):
    if parallel and n_jobs > 1:
        waveforms = Parallel(n_jobs=n_jobs)(delayed(calc_waveforms)(block_path)
                                            for block_path in morphs.paths.blocks())
    else:
        waveforms = [calc_waveforms(block_path) for block_path in morphs.paths.blocks()]
    waveforms = {k: v for k, v in waveforms}
    with open(morphs.paths.WAVEFORMS_PKL.as_posix(), 'wb') as f:
        pickle.dump(waveforms, f)


@morphs.utils.load._load(morphs.paths.WAVEFORMS_PKL, generate_waveforms)
def load_waveforms():
    '''
    returns dictionary such that:
    waveform_dict = morphs.load.waveforms()
    waveforms, cluster_map = waveform_dict[block_path]
    '''
    with open(morphs.paths.WAVEFORMS_PKL.as_posix(), 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    generate_waveforms()
