from __future__ import absolute_import
import numpy as np
import morphs
import pytest
from morphs.data.derivative import f_poly, p0_poly, fit_derivative


def test_f_poly():
    x = np.linspace(1, 128)
    assert f_poly(x, [-5, 0, -0.5, 1]).shape == x.shape
    x.reshape((-1, 1))
    assert f_poly(x, [-5, 0, -0.5, 1]).shape == x.shape
    x = np.array([1, 128])
    temp = f_poly(x, [0, 1])
    assert temp[0] == np.exp(-1), "centering not working"
    assert temp[1] == np.exp(1), "centering not working"


def test_fit_derivative():
    pair_df = morphs.load.pop_pair_df()
    morphs.data.parse.morph_dim(pair_df)
    for block_path, block_group in pair_df.groupby("block_path"):
        for morph_dim, morph_dim_group in block_group.groupby("morph_dim"):
            for order in range(8):
                p0, bounds = p0_poly(order)
                popt, pcov = fit_derivative(morph_dim_group, p0, bounds=bounds)
                assert len(popt) == order + 1
            break
        break


@pytest.mark.run(order=3)
def test_load_gen_derivative_dict():
    assert not morphs.paths.DERIVATIVE_PKL.exists()
    dd = morphs.load.derivative_dict()
    assert morphs.paths.DERIVATIVE_PKL.exists()
    assert len(dd) > 0
    for block in dd:
        assert len(dd[block]) == 24
