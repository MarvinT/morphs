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


def grid(behavior_df, subjects=morphs.subj.BEHAVE_SUBJS,
         row='lesser_dim', col='greater_dim', hue='subj',
         row_order=None, col_order=None,
         fit_reg=True, x_bins=7,
         x_label='Morph Position', y_label='P(greater response)',
         legend=True, legend_title='Subject', size=5,
         sub_title="{row_name}                  {col_name}",
         **kwargs):
    if row and row_order is None:
        row_order = np.sort(behavior_df[row].unique())
    if col and col_order is None:
        col_order = np.sort(behavior_df[col].unique())

    if hue is 'subj':
        color_order = [morphs.subj.BEHAVE_COLOR_MAP[subj] for subj in subjects]
        hue_order = subjects
    else:
        color_order = None
        hue_order = kwargs.pop('hue_order', None)

    g = sns.lmplot(x="morph_pos", y="greater_response", x_bins=x_bins,
                   row=row, col=col, hue=hue, data=behavior_df,
                   fit_reg=False,
                   row_order=row_order, col_order=col_order,
                   hue_order=hue_order,
                   palette=color_order,
                   legend=False, height=size, aspect=1, **kwargs)
    if fit_reg:
        g.map_dataframe(_4pl, 'morph_pos', 'greater_response')
    g = g.set_titles(sub_title)
    if legend:
        g = g.add_legend(title=legend_title)
    g = g.set(xlim=(1, 128), ylim=(0, 1), xticks=[], yticks=[0.0, 0.5, 1.0])
    g = g.set_axis_labels(x_label, y_label)
    return g
