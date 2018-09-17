import morphs


def test_gen_cluster_accuracies():
    morphs.data.accuracies.gen_cluster_accuracies()
    assert morphs.paths.ACCURACIES_PKL.exists()
