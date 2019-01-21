import logging

import numpy as np
import matplotlib.pyplot as plt


LOG = logging.getLogger(__name__)


def plot_density(calibrator, savefig=None, show=None):
    line_colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', ]

    plt.figure(figsize=(20,20), dpi=100)

    nclasses = len(calibrator.classes_)
    for icls in range(nclasses):
        x_target = np.arange(0, 1, .01)
        x_other = (1 - x_target) / (nclasses - 1)
        X = np.concatenate([
                np.repeat(x_other, icls).reshape(len(x_other), icls),
                x_target.reshape(len(x_target), 1),
                np.repeat(x_other, nclasses-icls-1).reshape(len(x_other), nclasses-icls-1),
            ], axis=1)

        p = calibrator.transform(X)
        plt.plot(x_target, p[:,icls], label=calibrator.classes_[icls], c=line_colors[icls])

    plt.legend()

    if savefig is not None:
        plt.savefig(savefig)
    if show or savefig is None:
        plt.show()

    plt.close()


def plot_lr_histogram(X, y, savefig=None, show=None):
    plt.figure(figsize=(10, 7), dpi=100)
    plt.xlabel('LR')
    plt.ylabel('fractie')
    plt.grid(True, axis='y', linestyle=':')
    plt.grid(True, axis='x', linestyle=':')

    rangesize = 4
    binsize = .5
    bins = np.arange(-rangesize - binsize/2., rangesize + binsize/2., binsize)

    p_target = X[np.arange(len(X)), y]

    plots = []
    labels = []
    weights = []
    for label, cnt in zip(*np.unique(y, return_counts=True)):
        p_cls = p_target[y == label]
        plots.append(np.log10(p_cls / (1-p_cls)))
        labels.append(label)
        weights.append(np.ones(cnt) / cnt)

    plt.hist(plots,
        label=labels,
        width=.25,  alpha=.5, density=True, stacked=True,
        range=(np.min(bins), np.max(bins)),
        bins=bins,
        weights=weights,
        )

    ticks = np.arange(-rangesize, rangesize+1, 1.)
    labels = [ str(f) if f < 1 else str(int(f)) for f in np.power(10, ticks) ]
    plt.xticks(ticks=ticks, labels=labels, rotation=0)

    plt.legend()
    if savefig is not None:
        plt.savefig(savefig)
    if show or savefig is None:
        plt.show()

    plt.close()
