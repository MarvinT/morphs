'''Collection of loading helper functions'''
from __future__ import absolute_import
from __future__ import print_function
import pickle
import functools
from google_drive_downloader import GoogleDriveDownloader as gdd


def _pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            return pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, 'rb') as f:
            return pickle.load(f, encoding='latin1')


def _download(dest_file, file_id):
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    gdd.download_file_from_google_drive(file_id=file_id,
                                        dest_path=dest_file.as_posix())


def _create(file_loc, gen_func, download_func=None):
    def decorator_load(func):
        @functools.wraps(func)
        def wrapper_load(*args, **kwargs):
            prefer_download = kwargs.pop('prefer_download', True)
            try:
                exists = file_loc.exists()
            except AttributeError:
                exists = file_loc(args[0]).exists()
            if not exists:
                if prefer_download and download_func:
                    print('downloading, alternatively set prefer_download=False to generate the data yourself')
                    download_func()
                else:
                    print('generating')
                    gen_func()
            return func()
        return wrapper_load
    return decorator_load
