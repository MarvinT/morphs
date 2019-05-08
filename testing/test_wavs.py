from __future__ import absolute_import
import morphs
import pytest


@pytest.mark.run(order=0)
def test_download_wavs():
    morphs.load.wavfile.download_wavs()
    assert morphs.paths.WAV_DIR.exists()
