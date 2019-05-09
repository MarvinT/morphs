from __future__ import absolute_import
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import itertools
import morphs
from morphs.data import xcor
from morphs.plot import morph_grid


def morph_viz(spikes, tau=0.01, stim_length=0.4, n_dim=50, smooth=False, transpose=False, **kwargs):
    xlabel = "Morph Position"
    ylabel = "Stimulus Duration (s)"
    if transpose:
        xlabel, ylabel = ylabel, xlabel
    g = morph_grid(
        spikes,
        _morph_viz,
        ylabel,
        xlabel=xlabel,
        map_kwargs={
            "tau": tau,
            "stim_length": stim_length,
            "n_dim": n_dim,
            "smooth": smooth,
            "transpose": transpose
        },
        **kwargs
    )
    g.set(yticks=[0.0, stim_length / 2, stim_length])
    return g


def _morph_viz(tau=0.01, stim_length=0.4, n_dim=50, smooth=False, transpose=False, **kwargs):
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
    if transpose:
        x, y = y, x
    if smooth:
        ax.tricontourf(x, y, z, 20)
    else:
        ax.tripcolor(x, y, z)


def morph_xcor_viz(spikes, tau=0.01, stim_length=0.4, n_dim=50, **kwargs):
    g = morph_grid(
        spikes,
        _morph_xcor_viz,
        "Morph Position",
        map_kwargs={"tau": tau, "stim_length": stim_length, "n_dim": n_dim},
        **kwargs
    )
    g.set(yticks=[])
    return g


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
    xyz = xcor.corrcoef_to_xyz_sf(grid, morph_pos_list)
    ax = plt.gca()
    ax.imshow(xcor.interpolate_grid(xyz))
