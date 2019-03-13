from __future__ import absolute_import
from __future__ import print_function
import morphs
import pandas as pd
import numpy as np
import itertools
from joblib import Parallel, delayed


def calculate_pair_df(X, labels, reduced=False):
    spects = morphs.load.morph_spectrograms()
    spect_reps = {'%s%s%03d' % (l, g, i): spects[l][g][i][:, :, 0]
                  for l in spects for g in spects[l] for i in spects[l][g]}

    label_df = pd.DataFrame(data={'stim_id': labels})
    morphs.data.parse.stim_id(label_df)

    df_list = []
    for morph_dim, group in label_df.groupby('morph_dim'):
        df_list.append(pd.DataFrame.from_records([(morph_dim, i1, i2) for i1, i2 in itertools.combinations(group.index.values, 2)],
                                                 columns=('morph_dim', 'lesser_index', 'greater_index')))
    pair_df = pd.concat(df_list, ignore_index=True)
    morphs.data.parse.morph_dim(pair_df)

    for index in ['greater', 'lesser']:
        pair_df[index + '_morph_pos'] = label_df.loc[pair_df[index + '_index'].values, 'morph_pos'].values

    pair_df['morph_dist'] = pair_df['greater_morph_pos'] - pair_df['lesser_morph_pos']
    pair_df['spect_euclidean_dist'] = (
        pair_df['morph_dim'].str.cat(pair_df['greater_morph_pos'].map(lambda x: '%03d' % (x))).map(spect_reps) -
        pair_df['morph_dim'].str.cat(pair_df['lesser_morph_pos'].map(lambda x: '%03d' % (x))).map(spect_reps)).apply(np.linalg.norm)
    pair_df['neural_euclidian_dist'] = np.linalg.norm(
        X[pair_df['greater_index'].values, :] -
        X[pair_df['lesser_index'].values, :], axis=1)
    if reduced:
        X_red = morphs.data.neurometric.logistic_dim_reduction(X, labels)
        pair_df['red_neural_euclidian_dist'] = np.linalg.norm(
            X_red[pair_df['greater_index'].values, :] -
            X_red[pair_df['lesser_index'].values, :], axis=1)

    for col in ['lesser_index', 'greater_index', 'lesser_dim', 'greater_dim']:
        del pair_df[col]
    for col in ['greater_morph_pos', 'lesser_morph_pos', 'morph_dist']:
        pair_df[col] = pair_df[col].astype('uint8')
    pair_df['morph_dim'] = pair_df['morph_dim'].astype('category')

    return pair_df


def calculate_pop_pair_df(block_path):
    good_clusters = morphs.data.accuracies.good_clusters(block_path)
    spikes = morphs.load.ephys_data(block_path, good_clusters=good_clusters)
    X, labels = morphs.spikes.create_neural_rep(spikes, max_id_len=5)
    pair_df = calculate_pair_df(X, labels, reduced=True)
    pair_df['block_path'] = block_path
    pair_df['block_path'] = pair_df['block_path'].astype('category')
    pair_df['block_path'].cat.set_categories(morphs.data.accuracies.good_recs(cluster_accuracies))
    return pair_df


def gen_pop_pair_df(parallel=True):
    if parallel:
        all_pairs = Parallel(n_jobs=morphs.parallel.N_JOBS)(delayed(calculate_pop_pair_df)(block)
                                                            for block in morphs.data.accuracies.good_recs(cluster_accuracies))
    else:
        all_pairs = [calculate_pop_pair_df(block)
                     for block in morphs.data.accuracies.good_recs(cluster_accuracies)]
    all_pairs_df = pd.concat(all_pairs, ignore_index=True)
    morphs.paths.POP_PAIR_PKL.parent.mkdir(parents=True, exist_ok=True)
    all_pairs_df.to_pickle(morphs.paths.POP_PAIR_PKL)


@morphs.utils.load._load(morphs.paths.POP_PAIR_PKL, gen_pop_pair_df)
def load_pop_pair_df():
    return pd.read_pickle(morphs.paths.POP_PAIR_PKL.as_posix())


if __name__ == '__main__':
    gen_pop_pair_df()
