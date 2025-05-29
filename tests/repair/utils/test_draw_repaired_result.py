from repair.utils import draw_repaired_result


def test_draw_repaired_result(repaired_keras_model_dir, targeted_data_dir):
    target_dir = targeted_data_dir / "negative"
    draw_repaired_result.run(
        model_dir=repaired_keras_model_dir,
        target_dir=target_dir,
    )

    assert (target_dir / "repaired.png").exists()
