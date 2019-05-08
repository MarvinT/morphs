from __future__ import absolute_import
import morphs
from morphs.load.wavfile import download_wavs
import pytest


@pytest.mark.run(order=0)
def test_download_wavs():
    download_wavs()
    assert morphs.paths.WAV_DIR.exists()


def test_load_wav():
    data, sr = morphs.load.wav("ac120")
