import json

from repair.utils import overlay_radar_charts


def test_overlay_bubble_chart(tmp_path):
    input_dir = tmp_path / "results"
    input_dir.mkdir()

    dummy_base_result = {
        "0": {"success": 25, "failure": 75, "score": 25.0},
        "1": {"success": 50, "failure": 50, "score": 50.0},
        "2": {"success": 75, "failure": 25, "score": 75.0},
        "3": {"success": 20, "failure": 80, "score": 20.0},
        "4": {"success": 95, "failure": 5, "score": 95.0},
    }

    (input_dir / "base").mkdir()
    with open(input_dir / "base" / "results.json", "w") as f:
        dict_sorted = sorted(dummy_base_result.items(), key=lambda x: x[0])
        json.dump(dict_sorted, f)

    dummy_overlay1 = {
        "0": {"success": 30, "failure": 70, "score": 30.0},
        "1": {"success": 60, "failure": 40, "score": 60.0},
        "2": {"success": 10, "failure": 90, "score": 10.0},
        "3": {"success": 80, "failure": 20, "score": 80.0},
        "4": {"success": 55, "failure": 45, "score": 55.0},
    }

    (input_dir / "overlay").mkdir()
    with open(input_dir / "overlay" / "results.json", "w") as f:
        dict_sorted = sorted(dummy_overlay1.items(), key=lambda x: x[0])
        json.dump(dict_sorted, f)

    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    overlay_radar_charts.run(
        input_dir=str(input_dir / "base"),
        input_dir_overlay=str(input_dir / "overlay"),
        output_dir=str(output_dir),
    )

    assert (output_dir / "radar.png").exists()
