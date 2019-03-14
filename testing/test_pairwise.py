from __future__ import absolute_import
import morphs
import pytest
import numpy as np


@pytest.mark.run(order=2)
def test_load_gen_pop_pair_df():
    assert len(morphs.paths.blocks()) > 0
    assert not morphs.paths.POP_PAIR_PKL.exists()
    df = morphs.load.pop_pair_df()
    assert morphs.paths.POP_PAIR_PKL.exists()
    assert df['block_path'].dtype.name == 'category'


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
