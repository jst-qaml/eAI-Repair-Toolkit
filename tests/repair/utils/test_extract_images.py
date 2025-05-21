import h5py

from repair.utils import extract_images


def test_extract_iamges(fashion_mnist_repair_data_dir):
    input_dir = fashion_mnist_repair_data_dir

    extract_images.run(
        input_dir=input_dir,
    )

    with h5py.File(input_dir / "repair.h5") as hf:
        num_images = hf["images"].shape[0]

    imgs_dir = input_dir / "images"

    assert len(list(imgs_dir.glob("*.ppm"))) == num_images
