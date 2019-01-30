from __future__ import absolute_import
import morphs
import pytest
from click.testing import CliRunner
import pandas as pd
import numpy as np


@pytest.mark.run(order=2)
def test_load_generate_load_neurometric_null_all():
    num_shuffles = 8
    assert not morphs.paths.num_shuffle_pkl(num_shuffles).exists()
    morphs.load.neurometric_null_all(num_shuffles)
    assert morphs.paths.num_shuffle_pkl(num_shuffles).exists()


@pytest.mark.run(order=3)
def test_combine_neuro_null_all():
    num_shuffles = 8
    all_samples_df = morphs.load.neurometric_null_all(num_shuffles)

    nshuffle_dir = morphs.paths.num_shuffle_dir(num_shuffles).glob('*.pkl')
    all_samples_df2 = pd.concat([morphs.utils.load._pickle(pkl_path)
                                 for pkl_path in nshuffle_dir])
    assert len(all_samples_df) == len(all_samples_df2)
    assert np.all(np.sort(all_samples_df['errors']) == np.sort(all_samples_df2['errors']))


@pytest.mark.run(order=3)
def test_load_generate_load_ks_df():
    num_shuffles = 8
    assert not morphs.paths.ks_df_pkl(num_shuffles).exists()
    morphs.load.ks_df(num_shuffles)
    assert morphs.paths.ks_df_pkl(num_shuffles).exists()


@pytest.mark.run(order=-1)
def test_neurometric_main():
    runner = CliRunner()
    result = runner.invoke(morphs.data.neurometric.null._main, ['--num_shuffles=8'])
