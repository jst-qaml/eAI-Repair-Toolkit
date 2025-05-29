from repair.utils import draw_radar_chart


def test_draw_radar_chart(pretrained_keras_model_dir, fashion_mnist_repair_data_dir):
    data_dir = fashion_mnist_repair_data_dir

    draw_radar_chart.run(
        input_dir=str(data_dir),
        model_dir=str(pretrained_keras_model_dir),
    )

    assert (data_dir / "results.json").exists()
    assert (data_dir / "radar.png").exists()
