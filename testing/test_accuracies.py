import morphs
import pytest


@pytest.mark.run(order=3)
def test_gen_cluster_accuracies():
    assert len(morphs.paths.blocks()) > 0
    morphs.data.accuracies.gen_cluster_accuracies()
    assert morphs.paths.ACCURACIES_PKL.exists()
