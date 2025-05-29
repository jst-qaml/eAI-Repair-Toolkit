"""Utility function:draw radar chart with subset."""

import json
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt


def run(
    *,
    input_dir: str,
    input_dir_overlay: str,
    input_dir_overlay2: str = None,
    output_dir: str = None,
    min_lim: int = 0,
    max_lim: int = 100,
    filename: str = "radar.png",
    legend: str = "Inputs (base)",
    legend_overlay: str = "Inputs (overlay)",
    legend_overlay2: str = "Inputs (overlay2)",
):
    """Overlay radar charts.

    Parameters
    ----------
    input_dir : str
        A path to the directory where 'result.json' for ground truth exists.
    input_dir_overlay : str
        A path to the directory where the 'result.json' for comparison exists.
    input_dir_overlay2 : str, default=None
        A path to the directory where the 'result.json' for extra comparison exists.
    output_dir : str, default=None
        A path to the directory where the chart will be generated.
        If it is None, the value of `input_dir` will be set.
    filename : str, default="radar.png"
        A file name of the generated image.
    min_lim : int, default=0
        Set the minimum radial axis view limit. This value will be passed to pyplot.
    max_lim : int, default=100
        Set the maximum radial axis view limit. This value will be passed to pyplot.
    legend : str, default="Inputs (base)"
        A legend for ground truth.
    legend_overlay : str, default="Inputs (overlay)"
        A legend for comparison data.
    legend_overlay2 : str, default="Inputs (overlay2)"
        A legend for extra comparison data.

    """
    if input_dir is None:
        raise ValueError("'input_dir' is required.")

    if input_dir_overlay is None:
        raise ValueError("'input_dir_overlay' is required.")

    if output_dir is None:
        output_dir = Path(input_dir)
    output_dir = Path(output_dir)

    fig = plt.figure()

    ax = _draw_each_radar_chart(input_dir, fig, min_lim, max_lim, legend, "o-")
    _draw_each_radar_chart(input_dir_overlay, fig, min_lim, max_lim, legend_overlay, "^-")

    if input_dir_overlay2 is not None:
        _draw_each_radar_chart(input_dir_overlay2, fig, min_lim, max_lim, legend_overlay2, "*-")

    if input_dir_overlay2 is None:
        legends = [legend, legend_overlay]
    else:
        legends = [legend, legend_overlay, legend_overlay2]
    ax.legend(legends, bbox_to_anchor=(-0.38, 1.15), loc="upper left", borderaxespad=0)

    fig.savefig(output_dir / filename, format="png", dpi=300)
    plt.close(fig)


def _draw_each_radar_chart(input_dir, fig, min_lim, max_lim, legend, plot):
    """Draw each radar chart.

    Parameters
    ----------
    input_dir : str
        same as the callee function.
    fig
        An instance of matplotlib.
    min_lim : int, default=0
        Set the minimum radial axis view limit.
    max_lim : int, default=100
        Set the maximum radial axis view limit.
    legend : str
        A legend for input data.
    plot: str
        A form of dots.

    """
    with open(Path(input_dir) / "results.json") as f:
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
