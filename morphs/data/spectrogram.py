import morphs
from google_drive_downloader import GoogleDriveDownloader as gdd


def load_morph_spectrograms():
    if not morphs.paths.SPECT_PKL.exists():
        download_all_loc()
    return morphs.data.load._pickle(morphs.paths.SPECT_PKL.as_posix())


def download_morph_spectrograms():
    morphs.paths.STIM_DIR.mkdir(parents=True, exist_ok=True)
    gdd.download_file_from_google_drive(file_id='1wirs8LQMSrc9jEaaI8P0oafMQs22Zs-l',
                                        dest_path=morphs.paths.SPECT_PKL.as_posix())
