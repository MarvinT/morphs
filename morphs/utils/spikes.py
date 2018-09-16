'''Utilities for spike processing'''
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
