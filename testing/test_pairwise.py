from __future__ import absolute_import
import morphs
import pytest


@pytest.mark.run(order=2)
def test_load_gen_pop_pair_df():
    assert len(morphs.paths.blocks()) > 0
    assert not morphs.paths.POP_PAIR_PKL.exists()
    df = morphs.load.pop_pair_df()
    assert morphs.paths.POP_PAIR_PKL.exists()
