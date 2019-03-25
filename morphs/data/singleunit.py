from __future__ import absolute_import
from __future__ import print_function
import morphs
import pickle
import numpy as np


def generate_single_unit_templates():
    psth = {}
    t = np.linspace(-.2, .6, 1000)

    for block_path in morphs.data.accuracies.good_recs():
        print(morphs.data.parse.blockpath_name(block_path))
        good_clusters = morphs.data.accuracies.good_clusters(block_path)
        spikes = morphs.load.ephys_data(
            block_path, good_clusters=good_clusters, collapse_endpoints=True)
        spikes['end'] = spikes['stim_id'].isin(list('abcdefghi'))
        psth[block_path] = {}
        for clusterID, cluster_group in spikes[spikes['end']].groupby('cluster'):
            psth[block_path][clusterID] = {}
            for motif, motif_group in cluster_group.groupby('stim_id'):
                X, _ = morphs.utils.spikes.create_neural_rep(motif_group, t=t)
                psth[block_path][clusterID][motif] = np.mean(X.T, axis=1)
    with open(morphs.paths.SINGLE_UNIT_TEMPLATES_PKL.as_posix(), 'wb') as f:
        pickle.dump(psth, f)


@morphs.utils.load._load(morphs.paths.SINGLE_UNIT_TEMPLATES_PKL, generate_single_unit_templates)
def load_single_unit_templates():
    return morphs.utils.load._pickle(morphs.paths.SINGLE_UNIT_TEMPLATES_PKL)


if __name__ == '__main__':
    generate_single_unit_templates()
