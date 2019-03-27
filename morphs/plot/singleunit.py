from __future__ import absolute_import
from __future__ import division
import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import morphs


def morph_viz(
    spikes,
    tau=0.01,
    stim_length=0.4,
    n_dim=50,
    smooth=False,
    title="",
    row_order="abcdef",
    col_order="cdefgh",
):
    g = sns.FacetGrid(
        spikes,
        col="greater_dim",
        row="lesser_dim",
        row_order=row_order,
        col_order=col_order,
    )
    g.map_dataframe(
        _morph_viz, tau=tau, stim_length=stim_length, n_dim=n_dim, smooth=smooth
    )
    g.set_titles("{row_name}     to     {col_name}")
    g.set_axis_labels("Morph Position", "Stimulus Duration (s)")
    if title:
        plt.subplots_adjust(top=0.95)
        g.fig.suptitle(title)
    g.despine(top=True, right=True, left=True, bottom=True)
    g.set(xticks=[], yticks=[0.0, stim_length / 2, stim_length])


def _morph_viz(tau=0.01, stim_length=0.4, n_dim=50, smooth=False, **kwargs):
    t = np.linspace(0, stim_length, n_dim)
    data = kwargs.pop("data")
    points = np.zeros((len(data["morph_pos"].unique()) * n_dim, 3))
    for i, (morph_pos, morph_pos_group) in enumerate(data.groupby("morph_pos")):
        trial_groups = morph_pos_group.groupby(["recording", "stim_presentation"])
        temp = (
            trial_groups["stim_aligned_time"]
            .apply(lambda x: morphs.spikes.filtered_response(x.values, tau=tau)(t))
            .mean()
        )
        points[i * n_dim: (i + 1) * n_dim, :] = np.array(
            list(zip(t, itertools.repeat(morph_pos), temp))
        )

    ax = plt.gca()
    x, y, z = (points[:, i] for i in range(3))
    if smooth:
        ax.tricontourf(y, x, z, 20)
    else:
        ax.tripcolor(y, x, z)


def morph_xcor_viz(
    spikes,
    tau=0.01,
    stim_length=0.4,
    n_dim=50,
    title="",
    row_order="abcdef",
    col_order="cdefgh",
):
    g = sns.FacetGrid(
        spikes,
        col="greater_dim",
        row="lesser_dim",
        row_order=row_order,
        col_order=col_order,
    )
    g.map_dataframe(_morph_xcor_viz, tau=tau, stim_length=stim_length, n_dim=n_dim)
    g.set_titles("{row_name}     to     {col_name}")
    g.set_axis_labels("Morph Position", "Morph Position")
    if title:
        plt.subplots_adjust(top=0.95)
        g.fig.suptitle(title)
    g.despine(top=True, right=True, left=True, bottom=True)
    g.set(xticks=[], yticks=[])


def _morph_xcor_viz(tau=0.01, stim_length=0.4, n_dim=50, **kwargs):
    t = np.linspace(0, stim_length, n_dim)
    data = kwargs.pop("data")
    grid = np.zeros((len(data["morph_pos"].unique()), n_dim))
    morph_pos_list = np.zeros(len(data["morph_pos"].unique()))
    for i, (morph_pos, morph_pos_group) in enumerate(data.groupby("morph_pos")):
        trial_groups = morph_pos_group.groupby(["recording", "stim_presentation"])
        grid[i, :] = (
            trial_groups["stim_aligned_time"]
            .apply(lambda x: morphs.spikes.filtered_response(x.values, tau=tau)(t))
            .mean()
        )
        morph_pos_list[i] = morph_pos
    xyz = morphs.data.xcor.corrcoef_to_xyz_sf(grid, morph_pos_list)
    ax = plt.gca()
    ax.imshow(morphs.data.xcor.interpolate_grid(xyz))
