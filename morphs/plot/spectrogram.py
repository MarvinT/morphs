import matplotlib.pylab as plt
import numpy as np


def single(dim, pos, spects, invert=True):
    im = spects[dim[0]][dim[1]][pos].astype(float)
    im /= np.max(im)
    if invert:
        im = 1 - im
    plt.figure()
    f = plt.imshow(im, aspect=0.5, extent=(0, 0.4, np.log(850), np.log(10000)))
    y_labels = [850, 1000, 2000, 4000, 10000]
    y_positions = [np.log(label) for label in y_labels]
    y_labels_str = ["850", "1k", "2k", "4k", "10k"]
    plt.yticks(y_positions, y_labels_str)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    return f.figure


def morph(dims, spects, divisions=16, fontsize=32, width=10, aspect=8, invert=True):
    morphs = {
        l
        + g: np.concatenate(
            [
                spects[l][g][pos].astype(float) / np.max(spects[l][g][pos])
                for pos in np.linspace(1, 128, divisions, dtype=int)
            ],
            axis=1,
        )
        for l, g in dims
    }
    for dim in morphs:
        if dim != dims[-1]:
            morphs[dim] = np.concatenate(
                [morphs[dim], np.ones((1, morphs[dim].shape[1], 3))], axis=0
            )
    im = np.concatenate([morphs[dim] for dim in dims], axis=0)
    if invert:
        im = 1 - im

    f = plt.figure(figsize=(width, float(width) * aspect * im.shape[0] / im.shape[1]))
    f = plt.imshow(im, aspect="auto")
    f.axes.get_xaxis().set_visible(False)

    l_labels, g_labels = zip(*(dim.upper() for dim in dims))

    ax_l = f.axes
    ax_g = ax_l.twinx()

    half_step = float(im.shape[0]) / 2 / len(l_labels)
    ax_l.yaxis.set_ticks(np.linspace(half_step, im.shape[0] - half_step, len(l_labels)))
    ax_l.yaxis.set_ticklabels(l_labels, fontsize=fontsize)
    ax_l.tick_params(axis="y", which="both", left=False, right=False)

    ax_g.set_ylim((-len(dims) + 0.5, 0.5))
    ax_g.yaxis.set_ticks(-np.arange(len(dims)))
    ax_g.yaxis.set_ticklabels(g_labels, fontsize=fontsize)
    ax_g.tick_params(axis="y", which="both", left=False, right=False)

    return f.figure
