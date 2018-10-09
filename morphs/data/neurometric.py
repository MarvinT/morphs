from __future__ import absolute_import
from __future__ import print_function
import pandas as pd
import numpy as np
import itertools
import pickle
import morphs
import sklearn as skl
import sklearn.linear_model
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed
from six.moves import range
from six.moves import zip


def hold_one_out_neurometric_fit_dist(representations, labels, behavior_subj, psychometric_params,
                                      shuffle_count=1024, parallel=True, n_jobs=morphs.parallel.N_JOBS):
    '''
    fits behavioral psychometric curves using the representation in a hold one out manner

    Parameters
    -----
    representations : np.array
        size = (num_data_points, num_dimensions)
    labels : iterable of string labels or np array of dtype='S5'
        labels : Pandas.DataFrame
            len = num_data_points
            required columns = ['morph_dim', 'morph_pos']
            overwritten/created columns = ['behave_data', 'p_r', 'p_l']
    behavior_subj : str
    shuffle_count : int
    calibrate : boolean

    Returns
    -----
    '''
    label_df = make_label_df(labels, behavior_subj, psychometric_params)
    behavior_df = make_behavior_df(behavior_subj, psychometric_params)

    if parallel:
        all_samples = Parallel(n_jobs=n_jobs)(delayed(calc_samples)(representations, label_df, behavior_df,
                                                                    idx, shuffle=shuffle) for idx, shuffle in [(i, i != 0) for i in range(shuffle_count + 1)])
    else:
        all_samples = [calc_samples(representations, label_df, behavior_df, idx, shuffle=shuffle)
                       for idx, shuffle in [(i, i != 0) for i in range(shuffle_count + 1)]]
    all_samples_df = pd.concat(all_samples, ignore_index=True)
    all_samples_df['subj'] = behavior_subj
    return all_samples_df


def hold_one_out_neurometric_fit_dist_all_subj(representations, labels, psychometric_params,
                                               shuffle_count=1024, parallel=True, n_jobs=morphs.parallel.N_JOBS):
    all_samples = []
    for subj in psychometric_params:
        print(subj)
        all_samples.append(hold_one_out_neurometric_fit_dist(representations, labels, subj, psychometric_params,
                                                             shuffle_count=shuffle_count, parallel=parallel,
                                                             n_jobs=n_jobs))
    return pd.concat(all_samples)


def make_label_df(labels, behavior_subj, psychometric_params):
    label_df = pd.DataFrame(data={'stim_id': labels})
    morphs.data.parse.stim_id(label_df)

    label_df['behave_data'] = False
    for dim, dim_group in label_df.groupby('morph_dim'):
        if dim in psychometric_params[behavior_subj]:
            label_df.loc[dim_group.index, 'behave_data'] = True

    morphs.data.parse.effective_morph(label_df, behavior_subj)
    return label_df


def make_behavior_df(behavior_subj, psychometric_params):
    morph_dims, morph_poss = list(zip(
        *itertools.product(list(psychometric_params[behavior_subj].keys()), np.arange(1, 129))))
    behavior_df = pd.DataFrame(data={'morph_dim': morph_dims, 'morph_pos': morph_poss})
    behavior_df['lesser_dim'] = behavior_df['morph_dim'].str[0]
    behavior_df['greater_dim'] = behavior_df['morph_dim'].str[1]
    morphs.data.parse.effective_morph(behavior_df, behavior_subj)
    for dim, dim_group in behavior_df.groupby('morph_dim'):
        psyc = morphs.logistic.normalized_four_param_logistic(
            psychometric_params[behavior_subj][dim])
        behavior_df.loc[dim_group.index, 'p_greater'] = dim_group['morph_pos'].apply(psyc)
    behavior_df['p_lesser'] = 1.0 - behavior_df['p_greater']
    behavior_df['p_left'], behavior_df['p_right'] = behavior_df[
        'p_lesser'], behavior_df['p_greater']
    behavior_df.loc[behavior_df['inverted'], 'p_right'] = behavior_df.loc[
        behavior_df['inverted'], 'p_lesser']
    behavior_df.loc[behavior_df['inverted'], 'p_left'] = behavior_df.loc[
        behavior_df['inverted'], 'p_greater']
    return behavior_df


def calc_samples(representations, label_df, behavior_df, idx, shuffle=False, tol=1e-4):
    error_list, dim_list = fit_held_outs(_merge_df(label_df, behavior_df, shuffle=shuffle),
                                         representations, tol=tol)
    return pd.DataFrame(data={'errors': error_list, 'held_out_dim': dim_list, 'shuffle_index': idx, 'shuffled': shuffle})


def _merge_df(label_df, behavior_df, shuffle=False):
    shuffle_effective_dim(label_df, shuffle=shuffle)
    shuffle_effective_dim(behavior_df, shuffle=False)
    return pd.merge(label_df, behavior_df[['shuffled_dim', 'effective_pos', 'p_left', 'p_right']],
                    on=['shuffled_dim', 'effective_pos'], how='left', validate='m:1')


def shuffle_effective_dim(df, shuffle=False):
    if shuffle:
        behave_dims = df[df['behave_data']]['effective_dim'].unique()
        non_behave_dims = set(df['effective_dim'].unique()) - set(behave_dims)
        dim_map = {dim: target for dim, target in zip(
            behave_dims, np.random.permutation(behave_dims))}
        dim_map.update({dim: dim for dim in non_behave_dims})
        df['shuffled_dim'] = df['effective_dim'].map(dim_map)
    else:
        df['shuffled_dim'] = df['effective_dim']


def fit_held_outs(merged_df, representations, accum='sse', tol=1e-4):
    mbdf = merged_df[merged_df['behave_data']]
    error_list = []
    dim_list = []
    for held_out_dim in mbdf['shuffled_dim'].unique():
        training_df = mbdf[mbdf['shuffled_dim'] != held_out_dim]
        held_out_df = mbdf[mbdf['shuffled_dim'] == held_out_dim]
        train_x = np.concatenate([representations[training_df.index, :],
                                  representations[training_df.index, :]])
        train_y = np.repeat([0, 1], len(training_df))
        train_weights = np.concatenate([training_df['p_left'], training_df['p_right']])

        test_x = representations[held_out_df.index, :]
        test_y = held_out_df['p_right']

        model = LogisticRegression(penalty='l2', tol=tol, warm_start=True).fit(
            train_x, train_y, sample_weight=train_weights)
        predicted_values = model.predict_proba(test_x)[:, 1]

        dim_list.append(held_out_dim)
        if accum == 'df':
            fit_df = held_out_df[['stim_id', 'p_right']].copy()
            fit_df['predicted'] = predicted_values
            error_list.append(fit_df)
        elif accum == 'mse':
            error_list.append(np.square(predicted_values - test_y).mean())
        elif accum == 'sse':
            error_list.append(np.square(predicted_values - test_y).sum())
        elif accum == 'sigmoid fit':
            raise NotImplementedError
        else:
            raise Exception('invalid accum option')
    return error_list, dim_list


def gen_held_out_df(merged_df, representations, melt=False):
    held_out_df = pd.concat(fit_held_outs(merged_df, representations, accum='df')[0])
    if melt:
        held_out_df = pd.melt(held_out_df, id_vars=['stim_id'], value_vars=['p_right', 'predicted'],
                              var_name='legend', value_name='p_right')
    morphs.data.parse.stim_id(held_out_df)
    return held_out_df


def logistic_dim_discriminants(X, labels):
    '''Returns a dictionary containing the logistic discriminating axis for the endpoints of each morph dimension'''
    dim_discriminants = {}
    labels = pd.Series(labels)
    morph_dims = labels.str[:2].unique()
    stim_ids, _ = morphs.data.parse.separate_endpoints(labels)
    motif_map = pd.DataFrame(stim_ids, columns=['motif']).groupby('motif')

    for morph_dim in morph_dims:
        lesser_dim, greater_dim = morph_dim
        endpoints_data = np.concatenate([X[motif_map.get_group(dim).index, :] for dim in morph_dim])
        endpoints_label = np.concatenate(
            [np.ones_like(motif_map.get_group(dim).index) * (dim == morph_dim[1]) for dim in morph_dim])
        model = LogisticRegression(penalty='l2')
        model.fit(endpoints_data, endpoints_label)
        dim_discriminants[morph_dim] = model.coef_

    return dim_discriminants


def logistic_dim_reduction(X, labels):
    '''Projects X onto the logistic discrminitating axis for the endpoints of each morph dimension'''
    dim_discriminants = logistic_dim_discriminants(X, labels)
    proj_matrix = np.array([dim_discriminants[dim] / np.linalg.norm(dim_discriminants[dim])
                            for dim in dim_discriminants]).squeeze().T
    return X.dot(proj_matrix)


def generate_neurometric_null_block(block_path, num_shuffles, cluster_accuracies, psychometric_params):
    nshuffle_dir = morphs.paths.num_shuffle_dir(num_shuffles)
    nshuffle_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = nshuffle_dir / (morphs.data.parse.blockpath_name(block_path) + '.pkl')

    good_clusters = morphs.data.accuracies.good_clusters(cluster_accuracies[block_path])
    spikes = morphs.data.load.ephys_data(
        block_path, good_clusters=good_clusters, shuffle_endpoints=True)

    X, labels = morphs.spikes.create_neural_rep(spikes, max_id_len=5)

    X_red = logistic_dim_reduction(X, labels)

    block_samples_df = hold_one_out_neurometric_fit_dist_all_subj(X_red, labels, psychometric_params,
                                                                  shuffle_count=num_shuffles, parallel=True,
                                                                  n_jobs=morphs.parallel.N_JOBS)
    block_samples_df['block'] = block_path
    block_samples_df.to_pickle(pkl_path.as_posix())


def load_neurometric_null_block(block_path, num_shuffles, cluster_accuracies, psychometric_params):
    nshuffle_dir = morphs.paths.num_shuffle_dir(num_shuffles)
    pkl_path = morphs.paths.num_shuffle_dir(
        num_shuffles) / (morphs.data.parse.blockpath_name(block_path) + '.pkl')
    if not pkl_path.exists():
        generate_neurometric_null_block(block_path, num_shuffles,
                                        cluster_accuracies, psychometric_params)
    with open(pkl_path.as_posix(), 'rb') as f:
        return pickle.load(f)


def generate_neurometric_null_all(num_shuffles):
    accuracies, cluster_accuracies = morphs.data.load.cluster_accuracies()
    psychometric_params = morphs.data.load.psychometric_params()
    all_samples = [load_neurometric_null_block(block_path, num_shuffles, cluster_accuracies, psychometric_params)
                   for block_path in morphs.data.accuracies.good_recs(cluster_accuracies)]
    all_samples_df = pd.concat(all_samples, ignore_index=True)
    all_samples_df.to_pickle(morphs.path.num_shuffle_pkl(num_shuffles))


def load_neurometric_null_all(num_shuffles):
    if not morphs.paths.num_shuffle_pkl(num_shuffles).exists():
        generate_neurometric_null_all(num_shuffles)
    with open(morphs.paths.num_shuffle_pkl(num_shuffles).as_posix(), 'rb') as f:
        return pickle.load(f)
