"""Utility function:draw radar chart with subset."""

import json
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt


def run(**kwargs):
    """Overlay radar charts."""
    if "input_dir" in kwargs:
        input_dir = Path(kwargs["input_dir"])
    else:
        raise TypeError("Require --input_dir")
    if "input_dir_overlay" in kwargs:
        input_dir_overlay = Path(kwargs["input_dir_overlay"])
    else:
        raise TypeError("Require --input_dir_overlay")
    if "input_dir_overlay2" in kwargs:
        input_dir_overlay2 = Path(kwargs["input_dir_overlay2"])
    else:
        input_dir_overlay2 = None
    if "output_dir" in kwargs:
        output_dir = Path(kwargs["output_dir"])
    else:
        output_dir = input_dir
    # For radar chart
    min_lim = kwargs["min_lim"] if "min_lim" in kwargs else 0
    max_lim = kwargs["max_lim"] if "max_lim" in kwargs else 100
    filename = kwargs["filename"] if "filename" in kwargs else r"radar.png"
    legend = kwargs["legend"] if "legend" in kwargs else "Inputs (base)"
    legend_overlay = kwargs["legend_overlay"] if "legend_overlay" in kwargs else "Inputs (overlay)"
    legend_overlay2 = (
        kwargs["legend_overlay2"] if "legend_overlay2" in kwargs else "Inputs (overlay2)"
    )

    # Instantiate plt.figure
    fig = plt.figure()

    # Plot
    ax = _draw_each_radar_chart(input_dir, fig, min_lim, max_lim, legend, "o-")
    _draw_each_radar_chart(input_dir_overlay, fig, min_lim, max_lim, legend_overlay, "^-")
    if input_dir_overlay2 is not None:
        _draw_each_radar_chart(input_dir_overlay2, fig, min_lim, max_lim, legend_overlay2, "*-")

    # Legend
    if input_dir_overlay2 is None:
        legends = [legend, legend_overlay]
    else:
        legends = [legend, legend_overlay, legend_overlay2]
    ax.legend(legends, bbox_to_anchor=(-0.38, 1.15), loc="upper left", borderaxespad=0)

    # Save radar chart
    fig.savefig(output_dir / filename, format="png", dpi=300)
    plt.close(fig)


def _draw_each_radar_chart(input_dir, fig, min_lim, max_lim, legend, plot):
    """Draw each radar chart.

    :param input_dir:
    :param fig:
    :param min_lim:
    :param max_lim:
    :param legend:
    :param plot:
    :return:
    """
    with open(input_dir / "results.json") as f:
        json_data = json.load(f)
        labels = []
        values = []
        for key in json_data:
            labels.append(key[0])
            values.append(key[1]["score"])
        values = np.concatenate((values, [values[0]]))
        angles = np.linspace(0, 2 * np.pi, len(labels) + 1, endpoint=True)

        ax = fig.add_subplot(111, polar=True)
        ax.plot(angles, values, plot)
        ax.fill(angles, values, alpha=0.2)
        ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels)
        ax.set_rlim(min_lim, max_lim)
        ax.legend(legend, loc=0)

        return ax
