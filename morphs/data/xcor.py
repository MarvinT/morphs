from __future__ import absolute_import
import numpy as np
import scipy as sp

"""
xcor dist types
xyz_vector_form: xyz
  [n*(n+1)/2, 3]
  may include non-zero "diagonal" elements
xyz_square_form: xyz
  [n*n, 3]
sp_square_form:
  [n], [n, n]
sp_vector_form: scipy vector_form
  [n], [n*(n-1)]
  sp_square_form[0], sp.spatial.distance.squareform(sp_square_form[1])
"""


def corrcoef_to_xyz_sf(grid, morph_pos_list):
    x, y = np.meshgrid(morph_pos_list, morph_pos_list)
    z = np.corrcoef(grid)
    return np.hstack((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)))


def df_to_xyz_vf(dist_df, dist_col="red_neural_cosine_dist"):
    return (
        dist_df.groupby(["lesser_morph_pos", "greater_morph_pos"])[dist_col]
        .agg(np.mean)
        .reset_index()
        .values
    )


def xyz_vf2sf(xyz):
    """xyz_vector_form to xyz_square_form"""
    non_diagonal_xyz = xyz[xyz[:, 0] != xyz[:, 1], :]
    return np.concatenate((xyz, non_diagonal_xyz[:, [1, 0, 2]]))


def df_to_xyz_sf(dist_df, dist_col="red_neural_cosine_dist"):
    return xyz_vf2sf(df_to_xyz_vf(dist_df, dist_col=dist_col))


def interpolate_grid(xyz):
    """
    Takes xyz_square_form data and interpolates it to full 128 spacing grid
    for use with ax.imshow(interpolated_grid)
    """
    grid_x, grid_y = np.mgrid[1:129, 1:129]
    return sp.interpolate.griddata(
        xyz[:, :2], xyz[:, 2], (grid_x, grid_y), method="nearest"
    )


def xyz_to_sp(xyz):
    "xyz_square_form to sp_square_form"
    morph_pos_list = np.unique(xyz[:, 0]).astype(int)
    morph_pos_map = {morph_pos: i for i, morph_pos in enumerate(morph_pos_list)}
    grid = np.zeros((len(morph_pos_list), len(morph_pos_list)))
    for i in range(xyz.shape[0]):
        x, y, z = xyz[i, :]
        grid[morph_pos_map[int(x)], morph_pos_map[int(y)]] = z
    return morph_pos_list, grid
