from google_drive_downloader import GoogleDriveDownloader as gdd
import morphs
import pytest


@pytest.mark.run(order=2)
def test_download_ephys_data():
    target = morphs.paths.EPHYS_DIR / 'B1096' / 'kwik' / 'Pen04_Lft_AP2500_ML50__Site02_Z2500__B1096_cat_P04_S02_1'
    target.mkdir(parents=True, exist_ok=True)
    dest_path = target / 'B1096_cat_P04_S02_1.kwik'
    gdd.download_file_from_google_drive(file_id='12bp8fHCC51PWOiX8QxziY7oM7sOxQetA',
                                        dest_path=str(dest_path))
    assert dest_path.exists()
    assert len(morphs.paths.blocks()) > 0
