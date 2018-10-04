'''Utilities for spike processing'''
from __future__ import absolute_import
import numpy as np


def filtered_response(spk_times, tau=.01):
    '''
    Returns function of the spike times convolved with a gaussian with width tau

    Parameters
    ------
    spk_times :  np.array
        np.array of the spike times
    tau : float, optional, default=.01
        width of the gaussian, exp(-t^2 / (2*tau^2))

    Returns
    ------
    f : function
        function that accepts np.array t
        and returns the instantaneous estimate of firing rate for each value of t
        uses O(len(spk_times)*len(t)) memory
    '''
    spk_times = spk_times.reshape((-1, 1))
    norm_factor = tau * np.sqrt(2. * np.pi)
    return lambda t: np.sum(np.exp(-(spk_times - t.reshape((1, -1))) ** 2 / (2 * tau * tau)), 0) / norm_factor


def create_neural_rep(spikes, stim_length=.4, max_id_len=None, num_samples=50):
    '''
    Returns a time-varying neural population representation for each stimuli presentation

    Parameters
    ------
    spikes : pandas dataframe
        a df containing columns cluster, stim_id, recording, stim_presentation, and stim_aligned_time
    stim_length : float
        the time in seconds of the length of all the stimuli
    max_id_len : int > 0 or None
        Max string length of the stim_ids
        If None (default) will calculate it.
    num_samples : int > 0
        number of time samples to use for each stimuli
        default = 50

    Returns
    ------
    X : np.array
        array of the neural rep
        shape = (number of stimuli presented, num_samples * number of spike cluster)
    labels: np.array
        array of the stim_id labels for each stimuli
        len = number of stimuli presented
    '''
    if max_id_len is None:
        spikes['stim_id'].str.len().max()
    clusters = spikes.cluster.unique()
    clust_map = {clust: i for i, clust in enumerate(clusters)}
    t = np.linspace(0, stim_length, num_samples)
    num_exemplars = len(spikes.groupby(('stim_id', 'recording', 'stim_presentation')))
    X = np.zeros((num_exemplars, num_samples * len(clusters)))
    labels = np.empty(num_exemplars, dtype='S%d' % (max_id_len))
    idx = 0
    for stim_id, stim_group in spikes.groupby('stim_id'):
        for (rec, stim_pres), trial_group in stim_group.groupby(['recording', 'stim_presentation']):
            cluster_groups = trial_group.groupby('cluster')
            temp = cluster_groups['stim_aligned_time'].apply(
                lambda x: filtered_response(x.values)(t))
            for i, (cluster, cluster_group) in enumerate(cluster_groups):
                X[idx, clust_map[cluster] *
                    num_samples:(clust_map[cluster] + 1) * num_samples] = temp.values[i]
            labels[idx] = stim_id
            idx += 1
    return X, labels
