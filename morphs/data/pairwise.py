from __future__ import absolute_import
from __future__ import print_function
import morphs
import pandas as pd
import numpy as np
import itertools
from joblib import Parallel, delayed
import click


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
    pair_df['neural_euclidian_dist'] = blocked_diff_norm(
        X,
        pair_df['greater_index'].values,
        pair_df['lesser_index'].values)
    if reduced:
        pair_df['red_neural_euclidian_dist'] = blocked_diff_norm(
            morphs.data.neurometric.logistic_dim_reduction(X, labels),
            pair_df['greater_index'].values,
            pair_df['lesser_index'].values)

    for col in ['lesser_index', 'greater_index', 'lesser_dim', 'greater_dim']:
        del pair_df[col]
    pair_df['morph_dim'] = pair_df['morph_dim'].astype('category')

    return pair_df


def blocked_norm(arr, block_size=2000, out=None):
    '''not used anymore... and these should go to utils somewhere...'''
    if out is None:
        ret = np.empty(arr.shape[0])
    else:
        ret = out
    for i in range(0, arr.shape[0], block_size):
        u = min(i + block_size, arr.shape[0])
        ret[i:u] = np.linalg.norm(arr[i:u], axis=1)
    return ret


def blocked_diff_norm(data, ind0, ind1, block_size=2000, out=None):
    if out is None:
        ret = np.empty(len(ind1))
    else:
        ret = out
    for i in range(0, len(ind1), block_size):
        u = min(i + block_size, len(ind1))
        ret[i:u] = np.linalg.norm(data[ind0[i:u], :] - data[ind1[i:u], :], axis=1)
    return ret


def calculate_pop_pair_df(block_path):
    good_clusters = morphs.data.accuracies.good_clusters(block_path)
    spikes = morphs.load.ephys_data(block_path, good_clusters=good_clusters)
    X, labels = morphs.spikes.create_neural_rep(spikes, max_id_len=5)
    pair_df = calculate_pair_df(X, labels, reduced=True)
    pair_df['block_path'] = block_path
    pair_df['block_path'] = pair_df['block_path'].astype('category')
    pair_df['block_path'].cat.set_categories(morphs.data.accuracies.good_recs())
    return pair_df


def gen_pop_pair_df(parallel=False, n_jobs=morphs.parallel.N_JOBS):
    if parallel and n_jobs > 1:
        all_pairs = Parallel(n_jobs=n_jobs)(delayed(calculate_pop_pair_df)(block)
                                            for block in morphs.data.accuracies.good_recs())
    else:
        all_pairs = [calculate_pop_pair_df(block)
                     for block in morphs.data.accuracies.good_recs()]
    all_pairs_df = pd.concat(all_pairs, ignore_index=True)
    all_pairs_df['block_path'] = all_pairs_df['block_path'].astype('category')

    morphs.paths.POP_PAIR_PKL.parent.mkdir(parents=True, exist_ok=True)
    all_pairs_df.to_pickle(morphs.paths.POP_PAIR_PKL)


@morphs.utils.load._load(morphs.paths.POP_PAIR_PKL, gen_pop_pair_df)
def load_pop_pair_df():
    return pd.read_pickle(morphs.paths.POP_PAIR_PKL.as_posix())


@click.command()
@click.option('--parallel', '-p', is_flag=True, help='whether to parallelize each block to its own process')
@click.option('--num_jobs', default=morphs.parallel.N_JOBS, help='number of parallel cores to use')
def _main(parallel, num_jobs):
    gen_pop_pair_df(parallel=parallel, n_jobs=num_jobs)


if __name__ == '__main__':
    _main()
