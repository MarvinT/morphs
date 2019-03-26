from __future__ import absolute_import
from __future__ import print_function
import pickle
import morphs


def calculate_psychometric_params(behavior_df):
    '''Fits psychometric params for each dim x each bird'''
    psychometric_params = {}
    for subj, subj_group in behavior_df.groupby('subj'):
        psychometric_params[subj] = {}
        for dim, dim_group in subj_group.groupby('morph_dim'):
            x = dim_group['morph_pos'].astype(float).values
            y = dim_group['greater_response'].astype(float).values
            psychometric_params[subj][dim] = morphs.logistic.fit_4pl(x, y, verbose=True)
    return psychometric_params


def generate_psychometric_params():
    '''
    Loads behavioral data, parses, calculates psychometric params,
    then drops them into a pkl file
    '''
    behavior_df = morphs.load.behavior_df()
    morphs.data.parse.stim_id(behavior_df)
    behavior_df = morphs.data.parse.behav_data_inverted(behavior_df)
    psychometric_params = calculate_psychometric_params(
        behavior_df[behavior_df['type_'] == 'normal'])
    morphs.paths.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(morphs.paths.PSYCHOMETRIC_PKL.as_posix(), 'wb') as f:
        pickle.dump(psychometric_params, f)


@morphs.utils.load._load(morphs.paths.PSYCHOMETRIC_PKL, generate_psychometric_params)
def load_psychometric_params():
    '''loads pickle file containing the fit psychometric parameters for each bird'''
    return morphs.utils.load._pickle(morphs.paths.PSYCHOMETRIC_PKL)


if __name__ == '__main__':
    generate_psychometric_params()
