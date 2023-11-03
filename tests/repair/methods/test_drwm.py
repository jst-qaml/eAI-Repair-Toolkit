import csv
import json
import shutil
from pathlib import Path

import pytest

from repair.core.model import load_model_from_tf
from repair.methods.drweightmerge.drwm import DRWeightMerge

RESOURCE_ROOT = Path("tests/resources/fashion-mnist")
TEST_MODEL = RESOURCE_ROOT / "model"


@pytest.fixture
def init_resource(tmp_path):
    resource_dir = shutil.copytree(RESOURCE_ROOT, tmp_path, dirs_exist_ok=True)
    return resource_dir


@pytest.fixture
def provide_setting(init_resource):
    setting = {
        "target_misclassifications": {
            "Pullover": ["Dress", "Shirt"],
            "Coat": ["Pullover"],
        },
        "general_misclassifications": ["Sandal"],
        "weights_precisions": {
            "Sandal": 0.4,
            "Pullover": 0.3,
            "Coat": 0.3,
        },
        "weights_misclassifications": {
            "Sandal": 0.3,
            "Pullover,Dress": 0.25,
            "Coat,Pullover": 0.25,
            "Pullover,Shirt": 0.2,
        },
    }
    path = init_resource / "dr_setting.json"
    with open(path, "w") as f:
        json.dump(setting, f)

    return init_resource, path


@pytest.fixture
def weights_snapshot(init_resource):
    negative_root = init_resource / "negative"
    with open(negative_root / "2" / "6" / "weights.csv", "w") as f:
        w = csv.writer(f)
        w.writerows(
            [
                [15, 297, 2, 0.13004746],
                [15, 501, 2, 0.16440994],
                [15, 292, 2, 0.23580645],
                [15, 297, 6, 0.26464906],
            ]
        )

    with open(negative_root / "2" / "3" / "weights.csv", "w") as f:
        w = csv.writer(f)
        w.writerows(
            [
                [15, 201, 2, 0.17066275],
                [15, 179, 2, 0.2591543],
                [15, 179, 3, 0.29003307],
            ]
        )

    with open(negative_root / "5" / "weights.csv", "w") as f:
        w = csv.writer(f)
        w.writerows(
            [
                [15, 450, 5, 0.1910668],
                [15, 274, 7, 0.27502546],
            ]
        )

    with open(negative_root / "4" / "2" / "weights.csv", "w") as f:
        w = csv.writer(f)
        w.writerows([[15, 373, 4, 0.28191236]])


@pytest.fixture
def model():
    return load_model_from_tf(TEST_MODEL)


@pytest.fixture
def dr():
    return DRWeightMerge()


def test_make_key(dr):
    result = dr._make_misclassification_dictkey("key")

    assert result == "key"


def test_make_keys(dr):
    result = dr._make_misclassification_dictkey("key1,key2")

    assert result == ("key1", "key2")


def test_load_setting(dr):
    setting = {
        "target_misclassifications": {
            "car": ["track", "bus"],
            "pedestrian": ["rider"],
        },
        "general_misclassifications": ["car", "track", "bus"],
        "weights_precisions": {
            "car": 0.1,
            "bus": 0.3,
        },
        "weights_misclassifications": {
            "car": 0.1,
            "pedestrian,rider": 0.2,
        },
    }
    dr._load_dr_setting(setting)

    assert dr.target_misclassifications == {
        ("car", "track"),
        ("car", "bus"),
        ("pedestrian", "rider"),
    }

    assert dr.general_misclassifications == ["car", "track", "bus"]

    assert dr.weights_precisions == pytest.approx(
        {
            "car": 1 / 4,
            "bus": 3 / 4,
        }
    )

    assert dr.weights_misclassifications == pytest.approx(
        {"car": 1 / 3, ("pedestrian", "rider"): 2 / 3}
    )


def test_load_setting_with_larger_values(dr):
    setting = {
        "target_misclassifications": {
            "car": ["track", "bus"],
            "pedestrian": ["rider"],
        },
        "general_misclassifications": ["car", "track", "bus"],
        "weights_precisions": {
            "car": 1,
            "bus": 3,
        },
        "weights_misclassifications": {
            "car": 1,
            "pedestrian,rider": 2,
        },
    }
    dr._load_dr_setting(setting)

    assert dr.weights_precisions == pytest.approx(
        {
            "car": 1 / 4,
            "bus": 3 / 4,
        }
    )

    assert dr.weights_misclassifications == pytest.approx(
        {"car": 1 / 3, ("pedestrian", "rider"): 2 / 3}
    )


def test_invalid_weights_precision_sum_zero(dr):
    setting = {
        "target_misclassifications": {},
        "general_misclassifications": [],
        "weights_precisions": {"car": 0.0, "bus": 0.0},
        "weights_misclassifications": {},
    }

    try:
        dr._load_dr_setting(setting)
        assert False
    except ValueError:
        assert True


def test_invalid_weights_precision_negative_value(dr):
    setting = {
        "target_misclassifications": {},
        "general_misclassifications": [],
        "weights_precisions": {"car": -0.1, "bus": 0},
        "weights_misclassifications": {},
    }

    try:
        dr._load_dr_setting(setting)
        assert False
    except ValueError:
        assert True


def test_invalid_weights_misclassification_sum_zero(dr):
    setting = {
        "target_misclassifications": {},
        "general_misclassifications": [],
        "weights_precisions": {},
        "weights_misclassifications": {"car": 0.0, "pedestrian,rider": 0.0},
    }

    try:
        dr._load_dr_setting(setting)
        assert False
    except ValueError:
        assert True


def test_invalid_weights_misclassification_negative_value(dr):
    setting = {
        "target_misclassifications": {},
        "general_misclassifications": [],
        "weights_precisions": {},
        "weights_misclassifications": {"car": 0, "pedestrian,rider": -0.6},
    }

    try:
        dr._load_dr_setting(setting)
        assert False
    except ValueError:
        assert True


class LocalizeTest:
    def test_localize(self, model, dr, provide_setting):
        resource_root, setting_path = provide_setting
        negative_root = resource_root / "negative"

        dr.set_options(
            **{
                "negative_root_dir": str(negative_root),
                "dataset": "fashion-mnist",
                "weight_path": str(setting_path),
            }
        )

        dr.localize(model=model, input_neg=None, output_dir=None)

        assert Path(negative_root / "2" / "3" / "weights.csv").exists()
        assert Path(negative_root / "2" / "6" / "weights.csv").exists()
        assert Path(negative_root / "4" / "2" / "weights.csv").exists()
        assert Path(negative_root / "5" / "weights.csv").exists()


class OptimizeTest:
    def test_optimize(self, model, dr, provide_setting, weights_snapshot):
        resource_root, setting_path = provide_setting
        negative_root = resource_root / "negative"

        dr.set_options(
            **{
                "negative_root_dir": str(negative_root),
                "dataset": "fashion-mnist",
                "weight_path": str(setting_path),
                "seed": 42,
                "num_particles": 2,
                "num_iterations": 2,
            }
        )

        input_neg = dr.load_input_neg(negative_root)
        input_pos = dr.load_input_pos(resource_root / "positive")

        dr.optimize(
            model=model,
            model_dir=None,
            input_neg=input_neg,
            input_pos=input_pos,
            output_dir=resource_root,
            weights=None,
        )

        assert Path.exists(resource_root / "repair")
        assert Path(resource_root / "repair").is_dir()

        assert Path.exists(resource_root / "weights.csv")
