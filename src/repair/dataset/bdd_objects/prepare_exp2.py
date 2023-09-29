"""Prepare BDD."""
import csv
from pathlib import Path

import h5py
import numpy as np
from tensorflow.keras.utils import to_categorical

from skimage import color, exposure, io, transform
from tqdm import tqdm


def prepare(root_path, output_path, divide_rate, random_state):
    """Prepare.

    :param root_path:
    :param output_path:
    :param divide_rate:
    :param random_state:
    :return:
    """
    orig_output_path = output_path
    for size in [32, 96, 224]:
        output_path = orig_output_path / str(size)
        output_path.mkdir(parents=True, exist_ok=True)
        print(
            f"=======Starting prepare with size {size}, "
            f"image_path {root_path} and output_path {output_path}======="
        )
        _get_images_and_labels(
            divide_rate,
            random_state,
            root_path,
            output_path,
            target_size_h=size,
            target_size_w=size,
        )


def _get_images_and_labels(
    divide_rate,
    random_state,
    img_path,
    output_path,
    target_size_h=96,
    target_size_w=96,
    classes=13,
    gray=False,
):
    dataset_name = str(img_path).split("/")[-1]

    # Get images
    all_image_paths = img_path.glob("images/*.png")
    all_image_paths = list(all_image_paths)
    all_image_paths = np.array(all_image_paths)
    np.random.default_rng(random_state).shuffle(all_image_paths)

    # Get labels
    all_labels = _get_labels(img_path / "labels.csv")

    save_images(
        all_image_paths,
        dataset_name,
        target_size_h,
        target_size_w,
        all_labels,
        output_path,
        classes,
    )


def save_images(
    paths,
    dataset_name,
    target_size_h,
    target_size_w,
    all_labels,
    output_path: Path,
    classes,
):
    """Save images.

    Parameters
    ----------
        paths
        dataset_name
        target_size_h
        target_size_w
        all_labels
        output_path: Path
        classes

    """
    missed_images_counter = 0
    chunk_size = 1000
    previous_size = 0
    with h5py.File(output_path / f"{dataset_name}.h5", "w") as hf:
        chunk_counter = 0
        images_list = []
        labels_list = []
        image_id_list = []
        attribute_id_list = []
        img_maxshape = None
        for image_path in tqdm(paths):
            # actually get the image and label
            img = _preprocess_img(io.imread(image_path), (target_size_h, target_size_w))
            image_id = image_path.name
            attribute_id = image_id.split("_")[-1]
            try:
                label = _get_train_class(all_labels, image_path)
            except KeyError as ke:
                print(ke)
                missed_images_counter += 1
                continue

            try:
                labels_list.append(to_categorical(label, num_classes=classes))
                images_list.append(img)
                image_id_list.append(image_id)
                attribute_id_list.append(attribute_id)

                chunk_counter += 1
                if (
                    chunk_counter == chunk_size or image_path == paths[-1]
                ):  # pass the list to the h5 file, so we never have a big list.
                    images_list = np.array(images_list)
                    labels_list = np.array(labels_list)
                    image_id_list = np.array(
                        image_id_list, dtype=h5py.special_dtype(vlen=str)
                    )
                    attribute_id_list = np.array(
                        attribute_id_list, dtype=h5py.special_dtype(vlen=str)
                    )

                    if (
                        img_maxshape is None
                    ):  # if it's the first time, initialize the h5py datasets
                        img_maxshape = (None,) + images_list.shape[1:]
                        lbl_maxshape = (None,) + labels_list.shape[1:]
                        imgid_maxshape = (None,) + image_id_list.shape[1:]
                        attrid_maxshape = (None,) + attribute_id_list.shape[1:]
                        imgs_dset = hf.create_dataset(
                            "images",
                            shape=images_list.shape,
                            maxshape=img_maxshape,
                            chunks=images_list.shape,
                            dtype=images_list.dtype
                            # , compression="gzip") may make ussage much slower
                        )
                        lbls_dset = hf.create_dataset(
                            "labels",
                            shape=labels_list.shape,
                            maxshape=lbl_maxshape,
                            chunks=labels_list.shape,
                            dtype=labels_list.dtype
                            # , compression="gzip")
                        )
                        imgids_dset = hf.create_dataset(
                            "image_ids",
                            shape=image_id_list.shape,
                            maxshape=imgid_maxshape,
                            chunks=image_id_list.shape,
                            dtype=image_id_list.dtype
                            # , compression="gzip"
                        )
                        attrids_dset = hf.create_dataset(
                            "attribute_ids",
                            shape=attribute_id_list.shape,
                            maxshape=attrid_maxshape,
                            chunks=attribute_id_list.shape,
                            dtype=attribute_id_list.dtype
                            # , compression="gzip"
                        )
                        imgs_dset[:] = images_list
                        lbls_dset[:] = labels_list
                        imgids_dset[:] = image_id_list
                        attrids_dset[:] = attribute_id_list
                    else:  # append to the datasets
                        # Resize the dataset to accommodate the next chunk of rows
                        imgs_dset.resize(previous_size + len(images_list), axis=0)
                        lbls_dset.resize(previous_size + len(labels_list), axis=0)
                        imgids_dset.resize(previous_size + len(image_id_list), axis=0)
                        attrids_dset.resize(
                            previous_size + len(attribute_id_list), axis=0
                        )

                        # Write the next chunk
                        imgs_dset[previous_size:] = images_list
                        lbls_dset[previous_size:] = labels_list
                        imgids_dset[previous_size:] = image_id_list
                        attrids_dset[previous_size:] = attribute_id_list

                    previous_size += len(images_list)
                    chunk_counter = 0
                    images_list = []
                    labels_list = []
                    image_id_list = []
                    attribute_id_list = []

            except TypeError as e:
                print(e)
                continue

        print(f"===========we missed {missed_images_counter} images============")
        print(dataset_name + "_images: {}".format(hf["images"].shape))
        print(dataset_name + "_labels: {}".format(hf["labels"].shape))
        print(dataset_name + "_image_ids: {}".format(hf["image_ids"].shape))
        print(dataset_name + "_attribute_ids: {}".format(hf["attribute_ids"].shape))


def _preprocess_img(img, target_size):
    # Rescale to target size
    img = transform.resize(img, target_size)

    # Histogram normalization in v channel
    hsv = color.rgb2hsv(img)
    hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
    img = color.hsv2rgb(hsv)

    return img


def _get_labels(file_path):
    """We only care about what label/class an image is. Ignore other information."""
    with open(file_path) as label_file:
        data = csv.DictReader(label_file)
        labels = {}

        for line in data:
            labels[line["#filename"]] = line["class_id"]

    return labels


def _get_train_class(labels, img_path):
    img_name = Path(img_path).name
    label_to_class = {
        "bicycle": 0,
        "bus": 1,
        "car": 2,
        "motorcycle": 3,
        "other person": 4,
        "other vehicle": 5,
        "pedestrian": 6,
        "rider": 7,
        "traffic light": 8,
        "traffic sign": 9,
        "trailer": 10,
        "train": 11,
        "truck": 12,
    }
    name = labels[img_name]
    if name.isdigit():
        # if the label is already the number instead of the name, just return it
        attribute = name
    else:
        attribute = label_to_class[name]
    return attribute
