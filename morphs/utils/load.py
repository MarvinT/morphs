'''Collection of loading helper functions'''
from __future__ import absolute_import
from __future__ import print_function
import pickle
import functools
from google_drive_downloader import GoogleDriveDownloader as gdd


def _pickle(pickle_file):
    '''
    Loads a pickle file that works in both py 2 and 3

    Parameters
    ------
    pickle_file : str
        path to pickle file to load
    '''
    try:
        with open(pickle_file, 'rb') as f:
            return pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, 'rb') as f:
            return pickle.load(f, encoding='latin1')


def _download(dest_file, file_id):
    '''
    returns a function that downloads file from Google Drive

    Parameters
    ------
    dest_file : pathlib Path
        destination to save the file
    file_id : str
        file_id for Google Drive file
    '''
    def download():
        gdd.download_file_from_google_drive(file_id=file_id,
                                            dest_path=dest_file.as_posix())
    return download


def _load(file_loc, gen_func, download_func=None):
    '''
    Creates a decorator that automatically downloads or generates processed data
    The wrapped function should just load and return the data

    Also memoizes the loaded data if the file is smaller than 1 Gb so that
    subsequent loading is instant.

    Parameters
    ------
    file_loc : pathlib Path or function that takes the first argument of the wrapped func and returns a Path
        where to save and load cached data from
    gen_func : function
        function that generates the data if it isn't cached
        accepts *args, **kwargs from wrapped func
    download_func : optional function
        function that downloads the data from Google Drive

    @wraps func
        Parameters
        ------
        prefer_download : boolean
            defaults to True
            will download before generating the file
        memoize : boolean
            defaults to iff cached_file.size < 1 Gb
            whether to store loaded data in memory 
            in case you are loading this file/data again
    '''
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
                filename.parent.mkdir(parents=True, exist_ok=True)
                if prefer_download and download_func:
                    print('downloading, alternatively set prefer_download=False to generate the data yourself')
                    download_func(*args, **kwargs)
                else:
                    print('generating')
                    gen_func(*args, **kwargs)
            small_file = filename.stat().st_size < 200 * 1024 ** 2  # 200 Mb
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
