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


def _load(file_loc, gen_func, download_func=None):
    def decorator_load(func):
        memoized_values = {}

        @functools.wraps(func)
        def wrapper_load(*args, **kwargs):
            prefer_download = kwargs.pop('prefer_download', True)
            try:
                exists = file_loc.exists()
                filename = file_loc
            except AttributeError:
                exists = file_loc(args[0]).exists()
                filename = file_loc(args[0])
            if not exists:
                if prefer_download and download_func:
                    print('downloading, alternatively set prefer_download=False to generate the data yourself')
                    download_func(*args, **kwargs)
                else:
                    print('generating')
                    gen_func(*args, **kwargs)
            small_file = filename.stat().st_size < 1024 ** 3  # 1 Gig
            memoize = kwargs.pop('memoize', small_file)
            if memoize:
                if filename in memoized_values:
                    return memoized_values[filename]
                else:
                    memoized_values[filename] = func(*args, **kwargs)
                    return memoized_values[filename]
            else:
                return func(*args, **kwargs)
        return wrapper_load
    return decorator_load
