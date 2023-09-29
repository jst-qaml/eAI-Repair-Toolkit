"""Utility for testing risk_calculation tool."""

from pathlib import Path

RESOURCE_DIR = Path(__file__).parents[1] / "resources/risk_calculation_tool"
SCALABEL_FORMAT_LABEL_PATH = RESOURCE_DIR / "bdd100k/labels/det_20/det_val.json"
CREATE_IMAGE_SUBSET_DIR = RESOURCE_DIR / "extracted_subset"
H5_DATASET_DIR = RESOURCE_DIR / "h5_dataset"
MODEL_DIR = Path(__file__).parents[1] / "resources/outputs_bdd_objects/model"

LABEL_TO_CLASS = [
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

TIMEOFDAY = ["morning", "night"]

SAMPLE_SIZE = len(LABEL_TO_CLASS) * len(TIMEOFDAY)
