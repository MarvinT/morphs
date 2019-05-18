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
    dpi=300,
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
            dpi=dpi,
        )


def cumulative_distribution(
    data,
    scaled=False,
    survival=False,
    label="Cumulative",
    fill=False,
    target_length=np.inf,
    downsample_method="geom",
    flip=False,
    preserve_ends=0,
    **kwargs
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
    x = np.concatenate([data, data[[-1]]])
    if len(x) > target_length:
        if downsample_method == "geom":
            x, y = log_downsample(x, y, target_length=target_length, flip=flip)
        elif downsample_method == "even_end":
            x, y = downsample(
                x, y, target_length=target_length, preserve_ends=preserve_ends
            )
        else:
            raise Exception("invalid downsample_method")
    plt.step(x, y, label=label, **kwargs)
    if fill:
        plt.fill_between(x, y, alpha=0.5, step="pre", **kwargs)


def downsample(x, y, target_length=1000, preserve_ends=0):
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    assert len(x) > target_length
    data = np.vstack((x, y))
    if preserve_ends > 0:
        l, data, r = np.split(data, (preserve_ends, -preserve_ends), axis=1)
    interval = int(data.shape[1] / target_length) + 1
    data = data[:, ::interval]
    if preserve_ends > 0:
        data = np.concatenate([l, data, r], axis=1)
    return data[0, :], data[1, :]


def geom_ind(stop, num=50):
    geo_num = num
    ind = np.geomspace(1, stop, dtype=int, num=geo_num)
    while len(set(ind)) < num - 1:
        geo_num += 1
        ind = np.geomspace(1, stop, dtype=int, num=geo_num)
    return np.sort(list(set(ind) | {0}))


def log_downsample(x, y, target_length=1000, flip=False):
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    assert len(x) > target_length
    data = np.vstack((x, y))
    if flip:
        data = np.fliplr(data)
    data = data[:, geom_ind(data.shape[1], num=target_length)]
    if flip:
        data = np.fliplr(data)
    return data[0, :], data[1, :]


def morph_grid(
    pair_df,
    map_func,
    ylabel,
    xlabel="Morph Position",
    map_kwargs={},
    title="",
    sub_title="{row_name}{col_name}",
    row_order="abcdef",
    col_order="cdefgh",
    sharey=True,
    sharex=True,
    **kwargs
):
    g = sns.FacetGrid(
        pair_df,
        col="greater_dim",
        row="lesser_dim",
        row_order=row_order,
        col_order=col_order,
        sharex=sharex,
        sharey=sharey,
        **kwargs
    )
    g.map_dataframe(map_func, **map_kwargs)
    g.set_titles(sub_title)
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


def format_morph_dim_label(g, row_order, col_order, morph_dims, flip=False, **kwargs):
    for row_index in range(len(row_order)):
        for col_index in range(len(col_order)):
            morph_dim = row_order[row_index] + col_order[col_index]
            if morph_dim in morph_dims:
                if flip:
                    morph_dim = morph_dim[::-1]
                format_morph_dim_ax_label(
                    g.axes[row_index, col_index], morph_dim=morph_dim, **kwargs
                )


def format_morph_dim_ax_label(ax, morph_dim="", x_axis=True, divisions=4):
    if x_axis:
        ax.set_xticks([1, 128])
        ax.set_xticklabels(morph_dim.upper())
        ax.set_xticks(
            np.linspace(1, 128, divisions, endpoint=False)[1:], minor=True
        )
    else:
        ax.set_yticks([1, 128])
        ax.set_yticklabels(morph_dim.upper())
        ax.set_yticks(
            np.linspace(1, 128, divisions, endpoint=False)[1:], minor=True
        )
