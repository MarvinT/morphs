from __future__ import absolute_import
import morphs
import pytest


@pytest.mark.run(order=0)
def test_download_ephys_data():
    dest_file = morphs.paths.EPHYS_DIR / 'B1096' / 'kwik' / \
        'Pen04_Lft_AP2500_ML50__Site02_Z2500__B1096_cat_P04_S02_1' / \
        'B1096_cat_P04_S02_1.kwik'
    morphs.utils.load._download(dest_file, '12bp8fHCC51PWOiX8QxziY7oM7sOxQetA')
    assert dest_file.exists()
    assert len(morphs.paths.blocks()) > 0
