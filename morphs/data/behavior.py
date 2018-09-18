import morphs
import pandas as pd
from google_drive_downloader import GoogleDriveDownloader as gdd


def reduce_behave_data(df):
    df = df[['class_', 'response', 'correct', 'rt', 'reward', 'stim_id', 'subj']]
    for col in ['correct', 'reward']:
        df[col] = df[col].astype(bool)
    for col in ['class_', 'response', 'subj']:
        df[col] = df[col].astype('category')
    return df


def gen_behavior_df():
    import behav
    from behav import loading
    behav_data = loading.load_data_pandas(morphs.subj.BEHAVE_SUBJS,
                                          morphs.paths.behave_data_folder())
    behavior_df = pd.concat([morphs.data.parse.behav_data_stim_id(behav_data[subj], subj) for
                             subj in morphs.subj.BEHAVE_SUBJS])
    behavior_df = reduce_behave_data(behavior_df)
    morphs.paths.BEHAVE_DIR.mkdir(parents=True, exist_ok=True)
    behavior_df.to_pickle(morphs.paths.BEHAVE_PKL.as_posix())


def load_behavior_df():
    if not morphs.paths.BEHAVE_PKL.exists():
        print('downloading beahvior_df, alternatively you could run' +
              'morphs.data.behavior.gen_behavior_df() to generate it')
        download_behavior_df()
    return pd.read_pickle(morphs.paths.BEHAVE_PKL.as_posix())


def download_behavior_df():
    morphs.paths.BEHAVE_DIR.mkdir(parents=True, exist_ok=True)
    gdd.download_file_from_google_drive(file_id='1wIOg1y0JpyeVgyDFrgOtzYv5XSymA-GX',
                                        dest_path=morphs.paths.BEHAVE_PKL.as_posix())
