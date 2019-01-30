from __future__ import absolute_import
import morphs
import pytest
from click.testing import CliRunner


@pytest.mark.run(order=2)
def test_load_generate_load_neurometric_null_all():
    num_shuffles = 8
    assert not morphs.paths.num_shuffle_pkl(num_shuffles).exists()
    morphs.data.neurometric.load_neurometric_null_all(num_shuffles)
    assert morphs.paths.num_shuffle_pkl(num_shuffles).exists()


@pytest.mark.run(order=-1)
def test_neurometric_main():
    runner = CliRunner()
    result = runner.invoke(morphs.data.neurometric.null._main, ['--num_shuffles=8'])
