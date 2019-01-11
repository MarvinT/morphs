'''Functions for loading and processing behavior data'''
from __future__ import absolute_import
from __future__ import print_function
import morphs
import pandas as pd


def reduce_behave_data(df):
    '''Trims df and changes dtypes to reduce pkl size'''
    df = df[['type_', 'class_', 'response', 'correct', 'rt', 'reward', 'stim_id', 'subj']]
    for col in ['correct', 'reward']:
        df[col] = df[col].astype(bool)
    for col in ['type_', 'class_', 'response', 'subj']:
        df[col] = df[col].astype('category')
    return df


def gen_behavior_df():
    '''Loads raw data and generates pkl containing all important behavior data'''
    import behav
    from behav import loading
    behav_data = loading.load_data_pandas(morphs.subj.BEHAVE_SUBJS,
                                          morphs.paths.behave_data_folder())
    behavior_df = pd.concat([morphs.data.parse.behav_data_stim_id(behav_data[subj], subj) for
                             subj in morphs.subj.BEHAVE_SUBJS])
    behavior_df = reduce_behave_data(behavior_df)
    morphs.paths.BEHAVE_DIR.mkdir(parents=True, exist_ok=True)
    behavior_df.to_pickle(morphs.paths.BEHAVE_PKL.as_posix())


def download_behavior_df():
    morphs.utils.load._download(morphs.paths.BEHAVE_PKL, '1wIOg1y0JpyeVgyDFrgOtzYv5XSymA-GX')


@morphs.utils.load._load(morphs.paths.BEHAVE_PKL, gen_behavior_df, download_func=download_behavior_df)
def load_behavior_df(prefer_download=True):
    '''Loads behavior df (and downloads it if it doesnt exist)'''
    return pd.read_pickle(morphs.paths.BEHAVE_PKL.as_posix())
