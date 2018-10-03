from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import scipy.optimize as op
from six.moves import range


def four_param_logistic(p):
    '''4p logistic function maker.

    Returns a function that accepts x and returns y for
    the 4-parameter logistic defined by p.

    The 4p logistic is defined by:
    y = A + (K - A) / (1 + exp(-B*(x-M)))

    Args:
        p: an iterable of length 4
            A, K, B, M = p

    Returns:
        A function that accepts a numpy array as an argument
        for x values and returns the y values for the defined 4pl curve.
    '''
    A, K, B, M = p

    def f(x):
        return A + (K - A) / (1 + np.exp(-B * (x - M)))
    return f


def normalized_four_param_logistic(p):
    '''removes the scaling of A and K'''
    A, K, B, M = p

    def f(x):
        return 1. / (1. + np.exp(-B * (x - M)))
    return f


def ln_like(p, x, y):
    '''log likelihood for fitting the four parameter logistic.

    Args:
        p: an iterable of length 4
            A, K, B, M = p
        x: a numpy array of length n
        y: a numpy array of length n
            must be of dtype double or float so multiplication works

    Returns:
        The log-likelihood that the samples y are drawn from a distribution
        where the 4pl(x; p) is the probability of getting y=1
    '''
    p_4pl = four_param_logistic(p)
    probs = p_4pl(x)
    return np.sum(y * np.log(probs) + (1 - y) * np.log(1 - probs))


def dln_like(p, x, y):
    '''gradient of the log likelihood for fitting the four parameter logistic.

    Args:
        p: an iterable of length 4
            A, K, B, M = p
        x: a numpy array of length n
        y: a numpy array of length n
            must be of dtype double or float so multiplication works

    Returns:
        The gradient of the log-likelihood that the samples y are drawn from
        a distribution where the 4pl(x; p) is the probability of getting y=1
    '''
    A, K, B, M = p

    def f(x):
        return A + (K - A) / (1 + np.exp(-B * (x - M)))

    def df(x):
        temp1 = np.exp(-B * (x - M))
        dK = 1. / (1. + temp1)
        dA = 1. - dK
        temp2 = temp1 / (1. + temp1) ** 2
        dB = (K - A) * (x - M) * temp2
        dM = -(K - A) * B * temp2
        return np.vstack((dA, dK, dB, dM))
    p_4pl = f(x)
    d_p_4pl = df(x)
    return np.sum(y * d_p_4pl / (p_4pl) - (1 - y) * d_p_4pl / (1 - p_4pl), 1)


def nll(*args):
    '''negative log-likelihood for fitting the 4 param logistic.'''
    return -ln_like(*args)


def ndll(*args):
    '''negative grad of the log-likelihood for fitting the 4 param logistic.'''
    return -dln_like(*args)


def est_pstart(x, y):
    '''basic estimation of a good place to start log likelihood maximization.

    Args:
        x: a numpy array of length n
            assumes a finite number of unique x values
        y: a numpy array of length n
            must be of dtype double or float so multiplication works

    Returns:
        p_start: an iterable of length 4 that should be a reasonable spot to
            start the optimization
            A, K, B, M = p_start
    '''
    p_start = [.01, .99, .2, 0]
    x_vals = np.unique(x)
    p_start[3] = np.mean(x_vals)
    y_est = np.array([np.mean(y[x == i]) for i in x_vals])
    midpoint_est = np.mean(np.where((y_est[0:-1] < .5) & (y_est[1:] >= .5)))
    if np.isnan(midpoint_est):
        return p_start
    p_start[3] = midpoint_est
    return p_start


def fit_4pl(x, y, p_start=None, verbose=False, epsilon=1e-16):
    '''Fits a 4 parameter logistic function to the data.

    Args:
        x: a numpy array of length n
            assumes a finite number of unique x values
        y: a numpy array of length n
            must be of dtype double or float so multiplication works
    optional:
        p_start: an iterable of length 4 that would be a reasonable spot to
            start the optimization. If None, tries to estimate it.
            A, K, B, M = p_start
            default=None
        verbose: boolean flag that allows printing of more error messages.
        epsilon: limits A and K between (epsilon, 1 - epsilon) for stability

    Returns:
        p_result: an iterable of length 4 that defines the model that
        is maximally likely
            A, K, B, M = p_result
    '''
    try:
        if not p_start:
            p_start = est_pstart(x, y)
    except TypeError:
        pass
    for i in range(3):
        if verbose and i > 0:
            print(('retry', i))
        result = op.minimize(nll, p_start, args=(x, y), jac=ndll, bounds=(
            (epsilon, 1 - epsilon), (epsilon, 1 - epsilon),
            (None, None), (None, None)))
        if result.success:
            return result.x
        else:
            if verbose:
                print((p_start, 'failure', result))
            p_start = result.x
    return False
