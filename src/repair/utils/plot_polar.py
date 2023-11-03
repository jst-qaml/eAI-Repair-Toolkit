"""Utility function:plot polar."""

import numpy as np

import matplotlib.pyplot as plt


def plot_polar(labels, values, path, min_lim, max_lim):
    """Plot polar.

    cf. https://qiita.com/1007/items/80406e098a4212571b2e

    Parameters
    ----------
    labels
    values
    path
    min_lim
    max_lim

    """
    angles = np.linspace(0, 2 * np.pi, len(labels) + 1, endpoint=True)
    values = np.concatenate((values, [values[0]]))
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, values, "o-")
    ax.fill(angles, values, alpha=0.25)
    ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels)
    ax.set_rlim(min_lim, max_lim)
    fig.savefig(path, format="png", dpi=300)
    plt.close(fig)
