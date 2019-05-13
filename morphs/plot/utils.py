import morphs
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns


def savefig(
    g,
    name,
    folder=morphs.paths.FIGURES_DIR,
    format=None,
    formats=["png", "pdf", "svg", "eps"],
    bbox_inches="tight",
    transparent=True,
    pad_inches=0,
):
    folder.mkdir(parents=True, exist_ok=True)
    if format:
        formats = [format]
    for format in formats:
        g.savefig(
            (folder / (name + "." + format)).as_posix(),
            format=format,
            bbox_inches=bbox_inches,
            transparent=transparent,
            pad_inches=pad_inches,
        )


def cumulative_distribution(
    data, scaled=False, survival=False, label="Cumulative", **kwargs
):
    """
    plots cumulative (or survival) step distribution
    """
    data = np.sort(data)
    if survival:
        data = data[::-1]
    y = np.arange(data.size + 1, dtype=float)
    if scaled:
        y /= y[-1]
    plt.step(np.concatenate([data, data[[-1]]]), y, label=label, **kwargs)


def morph_grid(
    pair_df,
    map_func,
    ylabel,
    xlabel="Morph Position",
    map_kwargs={},
    title="",
    row_order="abcdef",
    col_order="cdefgh",
    sharey=True,
    sharex=True,
):
    g = sns.FacetGrid(
        pair_df,
        col="greater_dim",
        row="lesser_dim",
        row_order=row_order,
        col_order=col_order,
        sharex=sharex,
        sharey=sharey,
    )
    g.map_dataframe(map_func, **map_kwargs)
    g.set_titles("{row_name}{col_name}")
    format_titles(g)
    g.set_axis_labels(xlabel, ylabel)
    if title:
        plt.subplots_adjust(top=0.95)
        g.fig.suptitle(title)
    g.despine(top=True, right=True, left=True, bottom=True)
    g.set(xticks=[])
    return g


def boundary(ax, morph_dim, color_dict=morphs.subj.BEHAVE_COLOR_MAP):
    """
    plots behavioral boundaries as vertical lines on provided ax
    """
    psychometric_params = morphs.load.psychometric_params()
    for bsubj in psychometric_params:
        if morph_dim in psychometric_params[bsubj]:
            ax.axvline(
                psychometric_params[bsubj][morph_dim][3],
                color=color_dict[bsubj],
                label=bsubj,
            )


def format_titles(g, upper=True, to_join=True):
    for ax in g.axes.flat:
        title = ax.get_title()
        if upper:
            title = title.upper()
        if to_join:
            title = "     to     ".join(title)
        ax.set_title(title)
