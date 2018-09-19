import morphs
import pytest


@pytest.mark.run(order=3)
def test_load_gen_cluster_accuracies():
    assert len(morphs.paths.blocks()) > 0
    assert not morphs.paths.ACCURACIES_PKL.exists()
    accuracies, cluster_accuracies = morphs.data.accuracies.load_cluster_accuracies()
    assert morphs.paths.ACCURACIES_PKL.exists()
