from __future__ import absolute_import
import morphs
import pytest
import numpy as np
import pandas as pd


@pytest.mark.run(order=1)
def test_load_gen_cluster_accuracies():
    assert len(morphs.paths.blocks()) > 0
    assert not morphs.paths.ACCURACIES_PKL.exists()
    accuracies, cluster_accuracies = morphs.data.accuracies.load_cluster_accuracies()
    assert morphs.paths.ACCURACIES_PKL.exists()


@pytest.mark.run(order=2)
def test_cluster_accuracy():
    assert len(morphs.paths.blocks()) > 0
    block_path = morphs.paths.blocks()[0]
    assert morphs.paths.ACCURACIES_PKL.exists()
    accuracies, cluster_accuracies = morphs.data.accuracies.load_cluster_accuracies()
    cluster = cluster_accuracies[block_path].index[-1]
    spikes = morphs.data.load.ephys_data(block_path, good_clusters=[cluster],
                                         collapse_endpoints=True)
    assert len(spikes['recording'].unique()) >= 1
    template_spikes = spikes[spikes['stim_id'].isin(list('abcdefgh'))]
    cluster_groups = template_spikes.groupby('cluster')

    morph_dims = spikes.morph_dim.unique()
    morph_dims = morph_dims[~pd.isnull(morph_dims)]
    morph_dims.sort()
    max_num_reps = np.max([len(stim_group.groupby(by=['recording', 'stim_presentation']))
                           for stim_id, stim_group in template_spikes.groupby('stim_id')])

    accuracies_list = [morphs.data.accuracies.cluster_accuracy(cluster, cluster_group, morph_dims, max_num_reps)
                       for (cluster, cluster_group) in cluster_groups]
