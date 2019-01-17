from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import pandas as pd
from ephys import core, rigid_pandas
import morphs


def calculate_ephys_data(block_path, good_clusters=None, collapse_endpoints=False, shuffle_endpoints=False):
    '''
    Loads ephys data and parses stimuli for this project

    Parameters
    ------
    block_path : str
    good_clusters : None or list of ints
        removes spikes that aren't from the provided list of cluster ids
    collapse_endpoints : boolean
        maps all of the morph endpoints on to the template name [a-h]
    shuffle_endpoints : boolean
        shuffles all of the stimuli labels of a each template
        not compatible with collapse_endpoints
    '''
    assert not (collapse_endpoints and shuffle_endpoints)
    spikes = core.load_spikes(block_path)

    if good_clusters is not None:
        spikes = spikes[spikes.cluster.isin(good_clusters)]

    stims = rigid_pandas.load_acute_stims(block_path)

    fs = core.load_fs(block_path)
    stims['stim_duration'] = stims['stim_end'] - stims['stim_start']
    rigid_pandas.timestamp2time(stims, fs, 'stim_duration')

    for rec, rec_group in stims.groupby('recording'):
        try:
            rec_group['stim_name'].astype(float)
            print('going to have to remove float stim recording ', rec)
            spikes = spikes[spikes['recording'] != rec]
            stims = stims[stims['recording'] != rec]
        except ValueError:
            if (rec_group['stim_duration'] > .41).any():
                print('removing long stim recording ', rec)
                spikes = spikes[spikes['recording'] != rec]
                stims = stims[stims['recording'] != rec]

    stim_ids = stims['stim_name'].str.decode('UTF-8')
    stim_ids = stim_ids.str.replace(r'_rec', '')
    stim_ids = stim_ids.str.replace(r'_rep\d\d', '')
    if collapse_endpoints:
        stim_ids, _ = morphs.data.parse.separate_endpoints(stim_ids)
    stims['stim_id'] = stim_ids
    morphs.data.parse.stim_id(stims)

    if shuffle_endpoints:
        end_stims = stims[(stims['morph_pos'] == 1) |
                          (stims['morph_pos'] == 128)]
        for morph_pos, morph_pos_group in end_stims.groupby('morph_pos'):
            if morph_pos == 1:
                end_stims.loc[morph_pos_group.index,
                              'end_stim'] = morph_pos_group['morph_dim'].str[0]
            elif morph_pos == 128:
                end_stims.loc[morph_pos_group.index,
                              'end_stim'] = morph_pos_group['morph_dim'].str[1]

        for end_stim, end_stim_group in end_stims.groupby('end_stim'):
            stims.loc[end_stim_group.index, 'stim_id'] = end_stim_group[
                'stim_id'].values[np.random.permutation(len(end_stim_group))]

    rigid_pandas.count_events(stims, index='stim_id')

    spikes = spikes.join(rigid_pandas.align_events(spikes, stims,
                                                   columns2copy=['stim_id', 'morph_dim', 'morph_pos',
                                                                 'stim_presentation', 'stim_start', 'stim_duration']))

    spikes['stim_aligned_time'] = (spikes['time_samples'].values.astype('int') -
                                   spikes['stim_start'].values)
    rigid_pandas.timestamp2time(spikes, fs, 'stim_aligned_time')

    return spikes


def ephys_data(block_path, good_clusters=None, collapse_endpoints=False, shuffle_endpoints=False, memoize=True):
    '''
    saves and loads processed ephys data if shuffle_endpoints==False for faster analysis
    shuffled data isn't saved so that each time you run it you can get a new sample

    Parameters
    ------
    block_path : str
    good_clusters : None or list of ints
        removes spikes that aren't from the provided list of cluster ids
    collapse_endpoints : boolean
        maps all of the morph endpoints on to the template name [a-h]
    shuffle_endpoints : boolean
        shuffles all of the stimuli labels of a each template
        not compatible with collapse_endpoints
    memoize : boolean
        whether to save the processed spikes dataframe for later use
    '''
    if shuffle_endpoints and memoize:
        print("I won't memoize shuffled data so each time you run it you can get a new sample.")
        memoize = False
    if not memoize:
        return calculate_ephys_data(block_path, good_clusters=good_clusters,
                                    collapse_endpoints=collapse_endpoints,
                                    shuffle_endpoints=shuffle_endpoints)

    file_loc = morphs.paths.ephys_pkl(block_path, collapse_endpoints)
    if file_loc.exists():
        spikes = pd.read_pickle(file_loc.as_posix())
    else:
        spikes = calculate_ephys_data(block_path, good_clusters=None,
                                      collapse_endpoints=collapse_endpoints,
                                      shuffle_endpoints=shuffle_endpoints)
        file_loc.parent.mkdir(parents=True, exist_ok=True)
        spikes.to_pickle(file_loc.as_posix())
    if good_clusters is not None:
        spikes = spikes[spikes.cluster.isin(good_clusters)]
    return spikes


if __name__ == '__main__':
    for block_path in morphs.paths.blocks():
        _ = ephys_data(block_path, collapse_endpoints=False)
        _ = ephys_data(block_path, collapse_endpoints=True)
