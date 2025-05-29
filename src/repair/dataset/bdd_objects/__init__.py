"""Berkley Deep Drive (BDD100K).

cf. https://bdd-data.berkeley.edu/index.html
"""
from __future__ import annotations

import h5py

import numpy as np
import ijson
from skimage import color, exposure, io, transform
from tqdm import tqdm
from pathlib import Path
from tensorflow.keras.utils import to_categorical

from repair.core import dataset

from . import prepare_exp2

__all__ = [
    "BDDObjects",
]


class BDDObjects(dataset.RepairDataset):
    """API for DNN with BDD."""

    @classmethod
    def get_name(cls) -> str:
        """Returns dataset name."""
        return "bdd-objects"

    @staticmethod
    def get_label_map() -> dict[str, int]:
        return {
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

    def set_extra_config(self, **kwargs):
        self.single_dir: bool = (
            kwargs["single_dir"] if kwargs.get("single_dir", None) is not None else False
        )
        # train:val:test:repair
        ratio = kwargs.get("data_ratio", None) or (
            0.5,
            0.1,
            0.2,
            0.2,
        )
        if sum(ratio) > 1:
            raise ValueError("Sum of 'data_ratio' must be 1.")

        self.train_ratio: float = ratio[0]
        self.val_ratio: float = ratio[1]
        self.test_ratio: float = ratio[2]
        self.repair_ratio: float = ratio[3]

    def _get_input_shape(self):
        """Set the input_shape and classes of BDD."""
        return (32, 32, 3), 13

    def prepare(self, input_dir: Path, output_dir: Path, divide_rate, random_state):
        """Prepare BDD objecets dataset.

        Parameters
        ----------
        input_dir : Path
        output_dir : Path
        divide_rate : float
        random_state : int, optional

        """
        # Make output directory if not exist
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        if "exp2" in str(input_dir):  # pragma: no cover
            prepare_exp2.prepare(input_dir, output_dir, divide_rate, random_state)
        else:
            self._get_images_and_labels(
                img_path=input_dir,
                output_path=output_dir,
                divide_rate=divide_rate,
                random_state=random_state,
            )

    def _get_images_and_labels(
        self,
        *,
        divide_rate,
        random_state,
        img_path,
        output_path,
        target_size_h=32,
        target_size_w=32,
        classes=13,
    ):
        if self.single_dir:
            repair_images = np.array(list(img_path.glob("*.jpg")))
            all_labels = self._get_labels(img_path / "image_info.json")
        else:
            train_image_paths = np.array(list(img_path.glob("train/*.jpg")))
            val_image_paths = np.array(list(img_path.glob("val/*.jpg")))
            repair_images = np.hstack((train_image_paths, val_image_paths))

            train_labels = self._get_labels(img_path / "train/image_info.json")
            val_labels = self._get_labels(img_path / "val/image_info.json")
            train_labels.update(val_labels)
            all_labels = train_labels

        np.random.default_rng(random_state).shuffle(repair_images)

        total = len(all_labels)
        val_start_index = int(total * self.train_ratio)
        test_start_index = int(total * self.val_ratio + val_start_index)
        repair_start_index = int(total * self.test_ratio + test_start_index)
        split_indices = [val_start_index, test_start_index, repair_start_index]

        train_imgs = repair_images[:val_start_index]
        val_imgs = repair_images[val_start_index:test_start_index]
        test_imgs = repair_images[test_start_index:repair_start_index]
        repair_imgs = repair_images[repair_start_index:] if self.repair_ratio != 0 else []

        if len(train_imgs) != 0:
            self.save_images(
                train_imgs,
                "train",
                target_size_h,
                target_size_w,
                all_labels,
                output_path,
                classes,
            )

        if len(val_imgs) != 0:
            self.save_images(
                val_imgs,
                "val",
                target_size_h,
                target_size_w,
                all_labels,
                output_path,
                classes,
            )

        if len(test_imgs) != 0:
            self.save_images(
                test_imgs,
                "test",
                target_size_h,
                target_size_w,
                all_labels,
                output_path,
                classes,
            )

        if len(repair_imgs) != 0:
            self.save_images(
                repair_imgs,
                "repair",
                target_size_h,
                target_size_w,
                all_labels,
                output_path,
                classes,
            )

    def save_images(
        self,
        paths,
        dataset_name,
        target_size_h,
        target_size_w,
        all_labels,
        output_path,
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
                img = self._preprocess_img(io.imread(image_path), (target_size_h, target_size_w))
                label = self._get_train_class(all_labels, image_path)
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
                            imgs_dset.resize(
                                (chunks_written * chunk_size) + len(images_list), axis=0
                            )
                            lbls_dset.resize(
                                (chunks_written * chunk_size) + len(labels_list), axis=0
                            )
                            imgids_dset.resize(
                                (chunks_written * chunk_size) + len(image_id_list), axis=0
                            )
                            attrids_dset.resize(
                                (chunks_written * chunk_size) + len(attribute_id_list),
                                axis=0,
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

    def _preprocess_img(self, img, target_size):
        # Rescale to target size
        img = transform.resize(img, target_size)

        # Histogram normalization in v channel
        hsv = color.rgb2hsv(img)
        hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
        img = color.hsv2rgb(hsv)

        return img

    def _get_labels(self, file_path):
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

    def _get_train_class(self, labels, img_path):
        img_name = Path(img_path).name
        label_to_class = BDDObjects.get_label_map()
        attribute = label_to_class[labels[img_name]]

        return attribute
