"""Utility function:draw radar chart with subset."""

import json

from tqdm import tqdm

from repair.core.model import test

from .plot_polar import plot_polar


def run(
    *,
    input_dir: str,
    output_dir: str,
    model_dir: str,
    filename: str = "radar.png",
    min_lim: int = 0,
    max_lim: int = 100,
):
    """Draw radar chart with labeled subset data.

    Parameters
    ----------
    input_dir : str
        A path to the directory where the 'labels.json' exists.
    output_dir : str
        A path to the directory where the chart will be generated. Default is same as 'input_dir'
    model_dir : str
        A path to the directory where the target model exists.
    filename : str, default="radar.png"
        A file name of the generated image.
    min_lim : int, default=0
        Set the minimum radial axis view limit. This value will be passed to pyplot.
    max_lim : int, default=100
        Set the maximum radial axis view limit. This value will be passed to pyplot.

    """
    if input_dir is None:
        raise ValueError("'input_dir' is required.")

    if output_dir is None:
        output_dir = input_dir

    if model_dir is None:
        raise ValueError("'model_dir' is required.")

    json_file = input_dir / "labels.json"
    with open(json_file) as fr:
        json_data = json.load(fr)
        labels = []
        values = []
        for key in tqdm(json_data, desc="Computing scores"):
            labels.append(key)
            data_dir = input_dir / str(key)
            if "score" in json_data[key]:
                values.append(json_data[key]["score"])
            elif (data_dir / "repair.h5").exists():
                score = test(model_dir, data_dir, "repair.h5")
                values.append(score[1] * 100)
                json_data[key]["score"] = score[1] * 100
            else:
                values.append(None)
        with open(json_file, "w") as fw:
            sorted_items = sorted(json_data.items(), key=lambda x: x[0])
            json.dump(sorted_items, fw, indent=4)
        plot_polar(labels, values, output_dir / filename, min_lim, max_lim)
