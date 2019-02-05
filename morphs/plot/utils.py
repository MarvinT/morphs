import morphs
import numpy as np
import matplotlib.pylab as plt


def savefig(g, name, folder=morphs.paths.FIGURES_DIR, format=None,
            formats=['png', 'pdf', 'svg', 'eps'],
            bbox_inches='tight', transparent=True, pad_inches=0):
    folder.mkdir(parents=True, exist_ok=True)
    if format:
        formats = [format]
    for format in formats:
        g.savefig((folder / (name + '.' + format)).as_posix(), format=format,
                  bbox_inches=bbox_inches, transparent=transparent, pad_inches=pad_inches)


def cumulative_distribution(data, scaled=False, survival=False, label='Cumulative', **kwargs):
    '''
    plots cumulative (or survival) step distribution
    '''
    data = np.sort(data)
    if survival:
        data = data[::-1]
    y = np.arange(data.size + 1, dtype=float)
    if scaled:
        y /= y[-1]
    plt.step(np.concatenate([data, data[[-1]]]), y, label=label, **kwargs)
