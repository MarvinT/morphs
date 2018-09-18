import cPickle as Pickle
import morphs


def calculate_psychometric_params(cumulative_data):
    psychometric_params = {}
    for subj, subj_group in cumulative_data.groupby('subj'):
        psychometric_params[subj] = {}
        for dim, dim_group in subj_group.groupby('morph_dim'):
            x = dim_group['morph_pos'].astype(float).values
            y = dim_group['greater_response'].astype(float).values
            psychometric_params[subj][dim] = morphs.logistic.fit_4pl(x, y, verbose=True)
    return psychometric_params


def generate_psychometric_params():
    cumulative_data = morphs.load.behavior_df()
    morphs.data.parse.stim_id(cumulative_data)
    cumulative_data = morphs.data.parse.behav_data_inverted(cumulative_data)
    psychometric_params = calculate_psychometric_params(cumulative_data)
    with open(morphs.paths.BEHAVE_PKL.as_posix(), 'wb') as f:
        Pickle.dump(psychometric_params, f)


def load_psychometric_params():
    with open(morphs.paths.BEHAVE_PKL.as_posix(), 'rb') as f:
        return Pickle.load(f)
