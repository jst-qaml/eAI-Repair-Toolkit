"""Utility function: calculate risk for exp2a."""

import json
import shutil
from collections import namedtuple
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf

from tqdm import tqdm

from repair.core.dataset import eai_dataset
from repair.core.model import load_model_from_tf

CALC_TARGETS = ["scene_prob", "miss_rate", "scene_prob_and_miss_rate"]

CommonArgs = namedtuple(
    "CommonArgs",
    [
        "calc_target",
        "h5_dataset_path",
        "create_image_subset_output_path",
        "output_dir",
        "format_json",
    ],
)
QueryArgs = namedtuple("QueryArgs", ["scalabel_format_label_path", "label", "attributes"])
PredictArgs = namedtuple("PredictArgs", ["model_dir"])

InputImage = namedtuple("InputImage", ["index", "label", "image_id", "attribute_ids"])
MissRateResult = namedtuple("MissRateResult", ["confusion_matrix", "misrecognisions"])


# Instead of given `dataset`, use `eAIDataset.EAIDataset`
def run(**kwargs):
    """Calculate risk.

    :param kwargs: Acceptable keys are written in docs/user_manual/README.adoc.
    """
    response = {}
    common_args = _get_common_args(**kwargs)
    input_images = _get_input_images(common_args.h5_dataset_path)
    image_id_to_path = _get_image_id_to_path(common_args.create_image_subset_output_path)

    if common_args.calc_target in [CALC_TARGETS[0], CALC_TARGETS[2]]:
        query_args = _get_query_args(**kwargs)
        response["query"] = _query_output_by_json(query_args)
        filtered_input_images = _get_filtered_input_images(input_images, query_args)
        response["scene_prob"] = _scene_prob_output_by_json(input_images, filtered_input_images)
        _generate_dataset_of_scene_prob(
            common_args.output_dir, image_id_to_path, filtered_input_images
        )
        input_images = filtered_input_images

    if common_args.calc_target in [CALC_TARGETS[1], CALC_TARGETS[2]]:
        predict_args = _get_predict_args(**kwargs)
        miss_rate_result = _calc_misrecognition_per_model(common_args, input_images, predict_args)
        response["miss_rate"] = _miss_rate_output_by_json(miss_rate_result)
        _generate_dataset_of_miss_rate(common_args.output_dir, image_id_to_path, miss_rate_result)

    with open(common_args.output_dir / "results.json", "w") as f:
        f.write(json.dumps({"response": response}, indent=4 if common_args.format_json else None))


# ==================================
# argument system


def _get_common_args(**kwargs):
    if "calc_target" in kwargs:
        calc_target = kwargs["calc_target"]
        if calc_target not in CALC_TARGETS:
            raise TypeError(f"Require: --calc_target is in {CALC_TARGETS}")
    else:
        raise TypeError("Require --calc_target")

    if "h5_dataset_path" in kwargs:
        h5_dataset_path = Path(kwargs["h5_dataset_path"])
    else:
        raise TypeError("Require --h5_dataset_path")

    if "create_image_subset_output_path" in kwargs:
        create_image_subset_output_path = Path(kwargs["create_image_subset_output_path"])
        if not create_image_subset_output_path.is_dir():
            raise TypeError("Require --create_image_subset_output_path must be directory")
    else:
        raise TypeError("Require --create_image_subset_output_path")

    if "output_dir" in kwargs:
        output_dir = Path(kwargs["output_dir"])
    else:
        output_dir = Path(r"outputs/")

    if "format_json" in kwargs:
        format_json = bool(kwargs["format_json"])
    else:
        format_json = False

    return CommonArgs(
        calc_target,
        h5_dataset_path,
        create_image_subset_output_path,
        output_dir,
        format_json,
    )


def _get_query_args(**kwargs):
    if "scalabel_format_label_path" in kwargs:
        scalabel_format_label_path = Path(kwargs["scalabel_format_label_path"])
    else:
        raise TypeError("Require --scalabel_format_label_path")

    if "label" in kwargs:
        label = kwargs["label"]
        try:
            label = str(int(label))
        except ValueError:
            try:
                label = str(label_to_class.index(label))
            except ValueError as e:
                raise ValueError(f"Invalid parameter: --label={label}") from e
    else:
        label = None

    if "attributes" in kwargs:
        attributes = {
            condition.split("=")[0].strip(): condition.split("=")[1].strip()
            for condition in kwargs["attributes"].split(",")
        }
    else:
        attributes = None

    return QueryArgs(scalabel_format_label_path, label, attributes)


def _get_predict_args(**kwargs):
    if "model_dir" in kwargs:
        model_dir = Path(kwargs["model_dir"])
    else:
        raise TypeError("Require --model_dir")

    return PredictArgs(model_dir)


# ==================================
# calculate system


def _get_input_images(h5_dataset_path):
    h5_paths = _expand_path(h5_dataset_path, "h5")
    input_images = []
    for h5_path in h5_paths:
        with h5py.File(h5_path, "r") as hf:
            labels = _convert_labels(hf["labels"][()])
            image_ids = [image_id.decode() for image_id in hf["image_ids"][()]]
            attribute_ids = [attribute_ids.decode() for attribute_ids in hf["attribute_ids"][()]]
            # Since python3.10 or later, assert is not necessary
            # because the `strict` option of zip can be used
            if not (len(labels) == len(image_ids) and len(labels) == len(attribute_ids)):
                raise ValueError(
                    """The number of images, labels and attributes
                                within the dataset must be the same."""
                )
            input_images.extend(
                [
                    InputImage((h5_path, index), *vals)
                    for index, vals in enumerate(zip(labels, image_ids, attribute_ids))
                ]
            )
    return input_images


def _expand_path(path, extension):
    return list(path.glob(f"*.{extension}")) if path.is_dir() else [path]


def _get_image_id_to_path(create_image_subset_output_path):
    print("reading folder of create_image_subset")
    image_id_to_path = {}
    total_size = 0
    src_dir = create_image_subset_output_path
    for image_path in src_dir.glob("*/*"):
        if image_path.is_file() and image_path.suffix != ".json":
            image_id_to_path[image_path.name] = image_path
            total_size += 1
    assert len(image_id_to_path) == total_size, "image id must be identical"
    print("finished reading")
    return image_id_to_path


def _convert_labels(labels):
    # https://github.com/jst-qaml/eAI-Repair/blob/527c9054ed2b68babf69b0621608bcee161f3845/dataset/utils/draw_bubble_chart.py#L47
    """Convert labels.

    convert labels from array representation into number,
    then reconvert them into string to set ticks properly.
    e.g.) [[0,0,...,1],...,[1,...,0]] => ["9",...,"0"]
    :param labels:
    """
    return np.array(list(map(lambda label: label.argmax(), labels)))


def _get_filtered_input_images(input_images, query_args):
    scalabel_paths = _expand_path(query_args.scalabel_format_label_path, "json")
    scalabel_labels = []
    for scalabel_path in scalabel_paths:
        with scalabel_path.open() as f:
            scalabel_labels.extend(json.load(f))
    image_attributes = {label["name"]: label["attributes"] for label in scalabel_labels}
    filtered_input_images = []
    for input_image in input_images:
        _, label, _, attribute_id = input_image
        if query_args.label and int(query_args.label) != label.item():
            continue
        if query_args.attributes and not all(
            [
                filter_key in image_attributes[attribute_id].keys()
                and image_attributes[attribute_id][filter_key] == filter_value
                for filter_key, filter_value in query_args.attributes.items()
            ]
        ):
            continue
        filtered_input_images.append(input_image)
    return filtered_input_images


def _calc_misrecognition_per_model(common_args, input_images, predict_args):
    print("start predict")
    model = load_model_from_tf(predict_args.model_dir)
    h5_paths = _expand_path(common_args.h5_dataset_path, "h5")
    matrix_size = len(label_to_class)

    confusion_matrix = [[0 for _ in range(matrix_size)] for _ in range(matrix_size)]
    misrecognisions = []
    progress_bar = tqdm(total=len(input_images))
    for h5_path in h5_paths:
        target_inputs = [
            input_image for input_image in input_images if input_image[0][0] == h5_path
        ]
        indices = [input_image[0][1] for input_image in target_inputs]
        # a little similar to dataset/tests.py
        dataset = EAIDatasetForRiskCalculationTool(str(h5_path))
        images_generator = dataset.get_generator_for_risk_calculation_tool(progress_bar, indices)
        results = model.predict(images_generator, verbose=0)
        if len(results) % dataset.batch_size != 0:
            raise ValueError("The number of inference results is different than expected")
        results = results[0 : len(indices)]
        pred_labels = _convert_labels(results)
        for input_image, pred in zip(target_inputs, pred_labels):
            _, truth, image_id, _ = input_image
            confusion_matrix[truth][pred] += 1
            if truth != pred:
                misrecognisions.append((image_id, truth.item(), pred.item()))
    print("finished predict")
    return MissRateResult(confusion_matrix, misrecognisions)


# ==================================
# output system


def _query_output_by_json(query_args):
    query = {}
    if query_args.label:
        query["label"] = query_args.label
    if query_args.attributes:
        for key, value in query_args.attributes.items():
            query[key] = value
    print("[Query]")
    print(json.dumps(query, indent=4))
    return query


def _scene_prob_output_by_json(input_images, filtered_input_images):
    total_size = len(input_images)
    filtered_size = len(filtered_input_images)
    summary = {
        "summary": {
            "total_image_count": total_size,
            "image_count_matched_to_query": filtered_size,
            "existence_rate": filtered_size / total_size,
        }
    }
    print("[Summary of scene prob]")
    print(json.dumps(summary["summary"], indent=4))
    return summary


def _generate_dataset_of_scene_prob(output_dir, image_id_to_path, filtered_input_images):
    dst_dir = output_dir / "matched_data"
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir()

    for input_image in filtered_input_images:
        image_id = input_image.image_id
        if image_id not in image_id_to_path:
            print(f"{image_id} is not found in create_image_subset_output")
            continue
        image_path = image_id_to_path[image_id]
        shutil.copyfile(image_path, dst_dir / image_path.name)


def _miss_rate_output_by_json(miss_rate_result):
    confusion_matrix, _ = miss_rate_result
    output = {
        "summary": {},
        "confusion_matrix": confusion_matrix,
    }
    for truth_class_id, predict_counts in enumerate(confusion_matrix):
        output["summary"][truth_class_id] = _generate_miss_rate_summary(
            truth_class_id, predict_counts
        )
    print("[Summary of miss rate]")
    print(json.dumps(output["summary"], indent=4))
    return output


def _generate_miss_rate_summary(truth_class_id, predict_counts):
    total = sum(predict_counts)
    correct = predict_counts[truth_class_id]
    return {
        "total_image_count": total,
        "misrecognized_image_count": total - correct,
        "misrecognized_rate": (total - correct) / total if total > 0 else 0,
    }


def _generate_dataset_of_miss_rate(output_dir, image_id_to_path, miss_rate_result):
    dst_dir = output_dir / "misrecognision_data"
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir()

    _, misrecognisions = miss_rate_result
    for truth, pred in set([(truth, pred) for _, truth, pred in misrecognisions]):
        (dst_dir / str(truth) / str(pred)).mkdir(parents=True)

    for image_id, truth, pred in misrecognisions:
        if image_id not in image_id_to_path:
            print(f"{image_id} is not found in create_image_subset_output")
            continue
        image_path = image_id_to_path[image_id]
        shutil.copyfile(image_path, dst_dir / str(truth) / str(pred) / image_path.name)


"""
based on
https://github.com/jst-qaml/eAI-Repair/blob/
8d5b0b47674f5ba9f20c52da048976deeec7c930/dataset/bdd_objects/prepare.py#L169
"""
label_to_class = [
    "bicycle",
    "bus",
    "car",
    "motorcycle",
    "other person",
    "other vehicle",
    "pedestrian",
    "rider",
    "traffic light",
    "traffic sign",
    "trailer",
    "train",
    "truck",
]


# ==================================
# extensions of dataset/eAIDataset


class EAIDatasetForRiskCalculationTool(eai_dataset.EAIDataset):
    """EAIDataset extends for risk_calculation_tool."""

    def get_generator_for_risk_calculation_tool(self, progress_bar, target_indices):
        """Return generator for risk_calculation_tool."""
        indices = np.array(target_indices)
        index_handler = eai_dataset.EAIIndexHandler(len(target_indices), self.batch_size, indices)
        return tf.data.Dataset.from_generator(
            EAIGeneratorForRiskCalculationTool(progress_bar, index_handler, self.hf["images"]),
            output_types=self.image_type,
            output_shapes=tf.TensorShape(
                [
                    self.batch_size,
                    self.image_shape[1],
                    self.image_shape[2],
                    self.image_shape[3],
                ]
            ),
        )


class EAIGeneratorForRiskCalculationTool(tf.keras.utils.Sequence):
    """EAIGenerator modified for risk_calculation_tool.

    Retrieves only `batch_size` elements of `index_handler` in order from the front,
    and returns the elements of that index in batches.
    According to the specification of `tf.data.Dataset.from_generator`,
    if the total number of indexes is not a multiple of `batch_size`,
    it is filled with the first element to be a multiple of batch_size.
    """

    def __init__(self, progress_bar, index_handler, dataset):
        """Initialize."""
        self.progress_bar = progress_bar
        self.index_handler = index_handler
        self.dataset = dataset

    def __call__(self):
        """Yield image batch."""
        batch_size = self.index_handler.batch_size
        total_size = len(self.index_handler.indices)
        for start in range(0, total_size, batch_size):
            indices = self.index_handler.indices[start : start + batch_size]
            self.progress_bar.update(len(indices))
            if self._is_contiguous(indices):
                batch = self.dataset[indices[0] : indices[0] + batch_size]
            else:
                batch = self.dataset[indices]
            for _ in range(batch_size):
                if len(batch) < batch_size:
                    batch = np.concatenate((batch, self.dataset[: batch_size - len(batch)]))
                else:
                    break
            yield batch

    def _is_contiguous(self, indices):
        for i in range(len(indices) - 1):
            if indices[i] + 1 != indices[i + 1]:
                return False
        return True

    def __len__(self):
        """Return index size."""
        return len(self.index_handler)
