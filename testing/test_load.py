from __future__ import absolute_import
import morphs
import pytest


@pytest.mark.run(order=2)
def test_load_ephys():
    assert len(morphs.paths.blocks()) > 0
    block_path = morphs.paths.blocks()[-1]
    assert morphs.paths.ACCURACIES_PKL.exists()
    accuracies, cluster_accuracies = morphs.data.accuracies.load_cluster_accuracies()
    cluster = cluster_accuracies[block_path].index[-1]
    spikes = morphs.load.ephys_data(block_path, good_clusters=[cluster])
    assert len(spikes) > 0


@pytest.mark.run(order=2)
def test_collapse_endpoints():
    assert len(morphs.paths.blocks()) > 0
    block_path = morphs.paths.blocks()[-1]
    assert morphs.paths.ACCURACIES_PKL.exists()
    accuracies, cluster_accuracies = morphs.data.accuracies.load_cluster_accuracies()
    cluster = cluster_accuracies[block_path].index[-1]
    spikes = morphs.load.ephys_data(block_path, good_clusters=[cluster],
                                    collapse_endpoints=True)
    assert len(spikes) > 0


@pytest.mark.run(order=2)
def test_shuffle_endpoints():
    assert len(morphs.paths.blocks()) > 0
    block_path = morphs.paths.blocks()[-1]
    assert morphs.paths.ACCURACIES_PKL.exists()
    accuracies, cluster_accuracies = morphs.data.accuracies.load_cluster_accuracies()
    cluster = cluster_accuracies[block_path].index[-1]
    spikes = morphs.load.ephys_data(block_path, good_clusters=[cluster],
                                    shuffle_endpoints=True)
    assert len(spikes) > 0
