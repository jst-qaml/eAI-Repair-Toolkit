"""Prepare BDD."""

import gc
from pathlib import Path

import h5py
import numpy as np
from tensorflow.keras.utils import to_categorical

import ijson
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
    _get_images_and_labels(divide_rate, random_state, root_path, output_path)


def _get_images_and_labels(
    divide_rate,
    random_state,
    img_path,
    output_path,
    target_size_h=32,
    target_size_w=32,
    classes=13,
    gray=False,
):
    # Get images
    train_image_paths = img_path.glob("train/*.jpg")
    train_image_paths = list(train_image_paths)
    train_image_paths = np.array(train_image_paths)
    val_image_paths = img_path.glob("val/*.jpg")
    val_image_paths = list(val_image_paths)
    val_image_paths = np.array(val_image_paths)
    all_image_paths = np.hstack((train_image_paths, val_image_paths))
    del val_image_paths
    del train_image_paths
    gc.collect()

    np.random.default_rng(random_state).shuffle(all_image_paths)

    # Get labels
    train_labels = _get_labels(img_path / "train/image_info.json")
    val_labels = _get_labels(img_path / "val/image_info.json")
    train_labels.update(val_labels)
    all_labels = train_labels
    del train_labels
    del val_labels
    gc.collect()

    train_test_division = 6  # the proportion of images to be used for test instead of training
    test_amount = len(all_image_paths) // train_test_division
    repair_amount = int((len(all_image_paths) - test_amount) * divide_rate)
    test_images_path = all_image_paths[:test_amount]
    repair_images_path = all_image_paths[test_amount : (test_amount + repair_amount)]
    train_images_path = all_image_paths[test_amount + repair_amount :]
    del all_image_paths
    gc.collect()

    save_images(
        test_images_path,
        "test",
        target_size_h,
        target_size_w,
        all_labels,
        output_path,
        classes,
    )
    del test_images_path
    gc.collect()
    save_images(
        repair_images_path,
        "repair",
        target_size_h,
        target_size_w,
        all_labels,
        output_path,
        classes,
    )
    del repair_images_path
    gc.collect()
    save_images(
        train_images_path,
        "train",
        target_size_h,
        target_size_w,
        all_labels,
        output_path,
        classes,
    )
    del train_images_path
    gc.collect()


def save_images(
    paths, dataset_name, target_size_h, target_size_w, all_labels, output_path, classes
):
    """Save images.

    Parameters
    ----------
        paths
        dataset_name
        target_size_h
        target_size_w
        all_labels
        output_path
        classes

    """
    chunk_size = 1000
    chunks_written = 0
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
            label = _get_train_class(all_labels, image_path)
            image_id = image_path.name
            attribute_id = image_id.split("_")[-1]

            try:
                labels_list.append(to_categorical(label, num_classes=classes))
                images_list.append(img)
                image_id_list.append(image_id)
                attribute_id_list.append(attribute_id)

                chunk_counter += 1
                if chunk_counter == chunk_size or image_path == paths[-1]:
                    # pass the list to the h5 file, so we never have a big list.
                    images_list = np.array(images_list)
                    labels_list = np.array(labels_list)
                    image_id_list = np.array(image_id_list, dtype=h5py.special_dtype(vlen=str))
                    attribute_id_list = np.array(
                        attribute_id_list, dtype=h5py.special_dtype(vlen=str)
                    )

                    if img_maxshape is None:
                        # if it's the first time, initialize the h5py datasets
                        img_maxshape = (None,) + images_list.shape[1:]
                        lbl_maxshape = (None,) + labels_list.shape[1:]
                        imgid_maxshape = (None,) + image_id_list.shape[1:]
                        attrid_maxshape = (None,) + attribute_id_list.shape[1:]
                        imgs_dset = hf.create_dataset(
                            "images",
                            shape=images_list.shape,
                            maxshape=img_maxshape,
                            chunks=images_list.shape,
                            dtype=images_list.dtype,
                            compression="gzip",
                        )
                        lbls_dset = hf.create_dataset(
                            "labels",
                            shape=labels_list.shape,
                            maxshape=lbl_maxshape,
                            chunks=labels_list.shape,
                            dtype=labels_list.dtype,
                            compression="gzip",
                        )
                        imgids_dset = hf.create_dataset(
                            "image_ids",
                            shape=image_id_list.shape,
                            maxshape=imgid_maxshape,
                            chunks=image_id_list.shape,
                            dtype=image_id_list.dtype,
                            compression="gzip",
                        )
                        attrids_dset = hf.create_dataset(
                            "attribute_ids",
                            shape=attribute_id_list.shape,
                            maxshape=attrid_maxshape,
                            chunks=attribute_id_list.shape,
                            dtype=attribute_id_list.dtype,
                            compression="gzip",
                        )
                        imgs_dset[:] = images_list
                        lbls_dset[:] = labels_list
                        imgids_dset[:] = image_id_list
                        attrids_dset[:] = attribute_id_list
                    else:  # append to the datasets
                        # Resize the dataset to accommodate the next chunk of rows
                        imgs_dset.resize((chunks_written * chunk_size) + len(images_list), axis=0)
                        lbls_dset.resize((chunks_written * chunk_size) + len(labels_list), axis=0)
                        imgids_dset.resize(
                            (chunks_written * chunk_size) + len(image_id_list), axis=0
                        )
                        attrids_dset.resize(
                            (chunks_written * chunk_size) + len(attribute_id_list), axis=0
                        )

                        # Write the next chunk
                        imgs_dset[chunks_written * chunk_size :] = images_list
                        lbls_dset[chunks_written * chunk_size :] = labels_list
                        imgids_dset[chunks_written * chunk_size :] = image_id_list
                        attrids_dset[chunks_written * chunk_size :] = attribute_id_list

                    chunk_counter = 0
                    chunks_written += 1
                    images_list = []
                    labels_list = []
                    image_id_list = []
                    attribute_id_list = []

            except TypeError:
                continue

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
    label_file = open(file_path)
    df = ijson.parse(label_file)
    labels = {}

    name = None
    attribute = None
    for prefix, _, value in df:
        if prefix == "images.item.name" and name is None:
            name = value
        if name is not None and prefix == "images.item.label":
            attribute = value
            labels[name] = attribute
            name = None
            attribute = None

    label_file.close()
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
    attribute = label_to_class[labels[img_name]]

    return attribute
