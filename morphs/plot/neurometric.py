import morphs
import numpy as np
import scipy as sp
import matplotlib.pylab as plt
import seaborn as sns


def _cf_4pl(x, A, K, B, M):
    return A + (K - A) / (1 + np.exp(-B * (x - M)))


def _4pl(x, y, color=None, **kwargs):
    data = kwargs.pop("data")

    popt, pcov = sp.optimize.curve_fit(
        _cf_4pl, data[x].values, data[y].values, maxfev=10000
    )
    try:
        result_4pl = morphs.logistic.four_param_logistic(popt)
        t = np.arange(128) + 1

        if color is None:
            lines, = plt.plot(x.mean(), y.mean())
            color = lines.get_color()
            lines.remove()

        plt.plot(t, result_4pl(t), color=color)
    except TypeError:
        pass


def held_out(labels, representations, behavior_subj, psychometric_params, **kwargs):
    label_df = morphs.data.neurometric.null.make_label_df(
        labels, behavior_subj, psychometric_params
    )
    behavior_df = morphs.data.neurometric.null.make_behavior_df(
        behavior_subj, psychometric_params
    )
    merged_df = morphs.data.neurometric.null._merge_df(label_df, behavior_df)
    held_out_df = morphs.data.neurometric.null.gen_held_out_df(
        merged_df, representations, melt=True
    )
    g = grid(held_out_df, **kwargs)
    return held_out_df, g


def grid(
    held_out_df,
    row_order=None,
    col_order=None,
    sup_title="",
    legend=True,
    legend_title="",
    p_right_leg_label="Behavioral (True) values",
    predicted_leg_label="Predicted values",
    sub_title="",
):
    held_out_df["legend"] = held_out_df["legend"].map(
        {"p_right": p_right_leg_label, "predicted": predicted_leg_label}
    )
    if row_order is None:
        row_order = np.sort(held_out_df["lesser_dim"].unique())
    if col_order is None:
        col_order = np.sort(held_out_df["greater_dim"].unique())
    g = sns.lmplot(
        x="morph_pos",
        y="p_right",
        hue="legend",
        col="greater_dim",
        row="lesser_dim",
        data=held_out_df,
        scatter=True,
        fit_reg=False,
        scatter_kws={"alpha": 0.3},
        row_order=row_order,
        col_order=col_order,
        legend=False,
        sharex=False,
    )
    g.map_dataframe(_4pl, "morph_pos", "p_right")
    if legend:
        g.add_legend(title=legend_title)
    g = g.set_titles(sub_title)
    morph_dims = held_out_df["morph_dim"].unique()
    morphs.plot.format_morph_dim_label(g, row_order, col_order, morph_dims)
    g.set(xlim=(0, 128), ylim=(0, 1), yticks=[0.0, 0.5, 1.0])
    g.set_axis_labels("Morph Position", "P(right response)")

    if sup_title:
        plt.subplots_adjust(top=0.95)
        g.fig.suptitle(sup_title)
    return g
