from repair.utils import overlay_bubble_chart


def test_overlay_bubble_chart(
    pretrained_keras_model_dir, repaired_keras_model_dir, fashion_mnist_repair_data_dir, tmp_path
):
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    overlay_bubble_chart.run(
        target_label=["0", "1", "2", "3", "4"],
        model_dir=pretrained_keras_model_dir,
        model_dir_overlay=repaired_keras_model_dir,
        test_dir=fashion_mnist_repair_data_dir,
        output_dir=str(output_dir),
    )

    assert (output_dir / "bubble.png").exists()
