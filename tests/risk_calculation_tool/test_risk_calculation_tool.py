"""Test for risk calculation tool."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from repair.utils.risk_calculation_tool import run as risk_calculation_tool

from .util import (
    CREATE_IMAGE_SUBSET_DIR,
    H5_DATASET_DIR,
    LABEL_TO_CLASS,
    MODEL_DIR,
    SAMPLE_SIZE,
    SCALABEL_FORMAT_LABEL_PATH,
    TIMEOFDAY,
)


class RiskCalculationToolTest:
    """Test class for risk calculation tool."""

    @pytest.fixture
    def _tmp_dir(self):
        with TemporaryDirectory() as dname:
            print(dname)
            yield Path(dname)

    # (filtering condition, matched_size)
    @pytest.fixture(
        params=[
            ({}, SAMPLE_SIZE),
            ({"attributes": f"timeofday={TIMEOFDAY[0]}"}, len(LABEL_TO_CLASS)),
            ({"label": LABEL_TO_CLASS[0]}, len(TIMEOFDAY)),
            ({"attributes": f"timeofday={TIMEOFDAY[0]}", "label": LABEL_TO_CLASS[0]}, 1),
        ]
    )
    def _filter_param(self, request):
        return request.param

    def test_scene_prob(self, _tmp_dir, _filter_param):
        """Test: calc_target=scene_prob."""
        risk_calculation_tool(
            **{
                "calc_target": "scene_prob",
                "h5_dataset_path": H5_DATASET_DIR / "test.h5",
                "create_image_subset_output_path": CREATE_IMAGE_SUBSET_DIR,
                "output_dir": _tmp_dir,
                "scalabel_format_label_path": SCALABEL_FORMAT_LABEL_PATH,
                **_filter_param[0],
            }
        )
        response = json.loads((_tmp_dir / "results.json").read_text())["response"]
        assert len(response["query"]) == len(_filter_param[0])
        summary = response["scene_prob"]["summary"]
        ideal_match_size = _filter_param[1]
        assert summary["total_image_count"] == SAMPLE_SIZE
        assert summary["image_count_matched_to_query"] == ideal_match_size
        assert summary["existence_rate"] == ideal_match_size / SAMPLE_SIZE
        assert len(list((_tmp_dir / "matched_data").glob("*.jpg"))) == ideal_match_size

    def test_miss_rate(self, _tmp_dir):
        """Test: calc_target=miss_rate."""
        risk_calculation_tool(
            **{
                "calc_target": "miss_rate",
                "h5_dataset_path": H5_DATASET_DIR / "test.h5",
                "create_image_subset_output_path": CREATE_IMAGE_SUBSET_DIR,
                "output_dir": _tmp_dir,
                "model_dir": MODEL_DIR,
                "format_json": "True",
            }
        )
        response = json.loads((_tmp_dir / "results.json").read_text())["response"]
        miss_rate = response["miss_rate"]
        all_summary = list(miss_rate["summary"].values())
        matrix = miss_rate["confusion_matrix"]
        assert len(all_summary) == len(matrix)
        assert sum(map(sum, matrix)) == SAMPLE_SIZE

    def test_generate_miss_rate_summary(self):
        """Test: function of _generate_miss_Rate_summary."""
        from repair.utils.risk_calculation_tool import _generate_miss_rate_summary

        truth_class_id = 1
        predict_counts = [0 for _ in range(len(LABEL_TO_CLASS))]
        predict_counts[0] = 1
        result = _generate_miss_rate_summary(truth_class_id, predict_counts)
        total = result["total_image_count"]
        miss_count = result["misrecognized_image_count"]
        miss_rate = result["misrecognized_rate"]
        assert total == sum(predict_counts)
        assert miss_rate == miss_count / total

    def test_scene_prob_and_miss_rate(self, _tmp_dir, _filter_param):
        """Test: calc_target=scene_prob_and_miss_rate."""
        risk_calculation_tool(
            **{
                "calc_target": "scene_prob_and_miss_rate",
                "h5_dataset_path": H5_DATASET_DIR / "test.h5",
                "create_image_subset_output_path": CREATE_IMAGE_SUBSET_DIR,
                "output_dir": _tmp_dir,
                "scalabel_format_label_path": SCALABEL_FORMAT_LABEL_PATH,
                "model_dir": MODEL_DIR,
                "format_json": "True",
                **_filter_param[0],
            }
        )
        response = json.loads((_tmp_dir / "results.json").read_text())["response"]
        assert len(response["query"]) == len(_filter_param[0])
        summary = response["scene_prob"]["summary"]
        ideal_match_size = _filter_param[1]
        assert summary["total_image_count"] == SAMPLE_SIZE
        assert summary["image_count_matched_to_query"] == ideal_match_size
        assert summary["existence_rate"] == ideal_match_size / SAMPLE_SIZE
        assert len(list((_tmp_dir / "matched_data").glob("*.jpg"))) == ideal_match_size

        miss_rate = response["miss_rate"]
        all_summary = list(miss_rate["summary"].values())
        matrix = miss_rate["confusion_matrix"]
        assert len(all_summary) == len(matrix)
        for summary, row in zip(all_summary, matrix):
            total = summary["total_image_count"]
            miss_count = summary["misrecognized_image_count"]
            miss_rate = summary["misrecognized_rate"]
            assert total == sum(row)
            assert miss_rate == (0 if total == 0 else miss_count / total)
        assert sum(map(sum, matrix)) == _filter_param[1]
