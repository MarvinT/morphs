from google_drive_downloader import GoogleDriveDownloader as gdd
import morphs
import librosa


def download_wavs():
    morphs.paths.WAV_ZIP.parent.mkdir(parents=True, exist_ok=True)
    gdd.download_file_from_google_drive(
        file_id="1cHZsDqxiiM1uXJM6Yt7VJmaBMCb_PsgJ", dest_path=morphs.paths.WAV_ZIP.as_posix(), unzip=True
    )


def load_wav(stim_name):
    if not morphs.paths.WAV_DIR.exists():
        download_wavs()
    file = morphs.paths.WAV_DIR / (stim_name + ".wav")
    assert file.exists()
    data, sr = librosa.core.load(file.as_posix(), sr=None)
    return data, sr
