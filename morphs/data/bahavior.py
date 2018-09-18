import morphs
import pandas as pd


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
    cumulative_data = pd.concat([morphs.data.parse.behav_data_stim_id(behav_data[subj], subj) for
                                 subj in morphs.subj.BEHAVE_SUBJS])
    cumulative_data = reduce_behave_data(cumulative_data)
    morphs.data.parse.stim_id(cumulative_data)
    cumulative_data.to_pickle(morphs.paths.BEHAVE_PKL.as_posix())


def load_behavior_df():
    pd.read_pickle(morphs.paths.BEHAVE_PKL.as_posix())
