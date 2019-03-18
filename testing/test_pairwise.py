from __future__ import absolute_import
import morphs
import pytest
import numpy as np
import scipy as sp
from scipy.spatial.distance import cosine


@pytest.mark.run(order=2)
def test_load_gen_pop_pair_df():
    assert len(morphs.paths.blocks()) > 0
    assert not morphs.paths.POP_PAIR_PKL.exists()
    df = morphs.load.pop_pair_df()
    assert morphs.paths.POP_PAIR_PKL.exists()
    assert df['block_path'].dtype.name == 'category'
    assert np.all(df['greater_morph_pos'] >= df['lesser_morph_pos'])


def test_calculate_pair_df():
    block_path = morphs.data.accuracies.good_recs()[-1]
    good_clusters = morphs.data.accuracies.good_clusters(block_path)
    spikes = morphs.load.ephys_data(block_path, good_clusters=good_clusters)
    X, labels = morphs.spikes.create_neural_rep(spikes, max_id_len=5)
    pair_df = morphs.data.pairwise.calculate_pair_df(X, labels, reduced=True, del_columns=False)
    for i in xrange(0, len(pair_df), 997):
        assert np.isclose(
            pair_df.loc[i, 'neural_cosine_dist'],
            cosine(
                X[pair_df.loc[i, 'lesser_index']],
                X[pair_df.loc[i, 'greater_index']],
            ),
        )


def test_blocked_norm():
    a = np.random.rand(100000, 100)
    assert np.allclose(
        np.linalg.norm(a, axis=1),
        morphs.data.pairwise.blocked_norm(a))


def test_blocked_diff_norm():
    test_size = 100000
    a = np.random.rand(test_size, 100)
    ind0 = np.random.permutation(test_size)
    ind1 = ind0[int(test_size / 2):]
    ind0 = ind0[:int(test_size / 2)]
    assert np.allclose(
        np.linalg.norm(a[ind0, :] - a[ind1, :], axis=1),
        morphs.data.pairwise.blocked_diff_norm(a, ind0, ind1))
