import morphs
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def _4pl(x, y, color=None, **kwargs):
    data = kwargs.pop("data")

    result = morphs.logistic.fit_4pl(data[x].values, data[y].values.astype(np.double))
    try:
        result_4pl = morphs.logistic.four_param_logistic(result)
        t = np.arange(128) + 1

        if color is None:
            lines, = plt.plot(x.mean(), y.mean())
            color = lines.get_color()
            lines.remove()

        plt.plot(t, result_4pl(t), color=color)
    except TypeError:
        pass


def grid(
    behavior_df,
    subjects=morphs.subj.BEHAVE_SUBJS,
    row="lesser_dim",
    col="greater_dim",
    hue="subj",
    row_order=None,
    col_order=None,
    fit_reg=True,
    x_bins=7,
    x_label="Morph Position",
    y_label="P(aligned response)",
    legend=True,
    legend_title="Subject",
    size=5,
    aspect=1,
    sub_title="",
    scatter_kws={"s": 1},
    **kwargs
):
    if row and row_order is None:
        row_order = np.sort(behavior_df[row].unique())
    if col and col_order is None:
        col_order = np.sort(behavior_df[col].unique())

    if hue is "subj":
        color_order = [morphs.subj.BEHAVE_COLOR_MAP[subj] for subj in subjects]
        hue_order = subjects
    else:
        color_order = None
        hue_order = kwargs.pop("hue_order", None)

    g = sns.lmplot(
        x="morph_pos",
        y="greater_response",
        x_bins=x_bins,
        row=row,
        col=col,
        hue=hue,
        data=behavior_df,
        fit_reg=False,
        row_order=row_order,
        col_order=col_order,
        hue_order=hue_order,
        palette=color_order,
        legend=False,
        height=size,
        aspect=aspect,
        scatter_kws=scatter_kws,
        sharex=False,
        **kwargs
    )
    if fit_reg:
        g.map_dataframe(_4pl, "morph_pos", "greater_response")
    g = g.set_titles(sub_title)
    morphs.plot.format_titles(g)
    morph_dims = behavior_df["morph_dim"].unique()
    if row is "lesser_dim" and col is "greater_dim":
        morphs.plot.format_morph_dim_label(g, row_order, col_order, morph_dims)
    elif col is "morph_dim":
        for col_index, morph_dim in enumerate(col_order):
            ax = g.axes.flatten()[col_index]
            ax.set_xticks([1, 128])
            ax.set_xticklabels(morph_dim.upper())
    else:
        print("I'll need to do something about these axis labels")
    if legend:
        g = g.add_legend(title=legend_title)
    g = g.set(xlim=(1, 128), ylim=(0, 1), yticks=[0.0, 0.5, 1.0])
    g = g.set_axis_labels(x_label, y_label)
    return g
