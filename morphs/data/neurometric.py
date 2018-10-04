import pandas as pd
import numpy as np
import sklearn as skl
import sklearn.linear_model
from sklearn.linear_model import LogisticRegression


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
