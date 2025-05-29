"""Utility function: extract images from h5 file."""

import shutil

import h5py

import imageio


def run(*, input_dir: str, output_dir: str = None, target_data: str = "repair.h5"):
    """Extract images from hdf5.

    This utility extract images and saves to `{output_dir}/images` as ppm.

    Notes
    -----
    This utility uses imageio library. This library requires the images more than one channels.

    Parameters
    ----------
    input_dir : str
        A path to the directory where 'target_data' exists.
    output_dir : str|None, default=None
        A path to the root directory where the generated image will be saved.
        If it is None, the value of `input_dir` will be set.
    target_data : str, default="repair.h5"
        A file name of the dataset. It must be hdf5 format.

    """
    if input_dir is None:
        raise ValueError("'input_dir' is required.")

    if output_dir is None:
        output_dir = input_dir

    if not target_data.endswith(".h5"):
        target_data = f"{target_data}.h5"

    output_dir = output_dir / "images"
    try:
        output_dir.mkdir(parents=True)
    except FileExistsError:
        shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)

    with h5py.File(input_dir / target_data, "r") as hf:
        images = hf["images"][()]
        index = 0
        for image in images:
            filename = f"{index:08}.ppm"
            imageio.imwrite(output_dir / filename, image)
            index += 1
