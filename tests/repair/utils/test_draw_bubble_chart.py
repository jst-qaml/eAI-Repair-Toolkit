import pytest

from repair.utils import draw_bubble_chart

pytestmark = pytest.mark.usefixtures("fix_seed")


def test_draw_radar_chart(pretrained_keras_model_dir, fashion_mnist_repair_data_dir):
    data_dir = fashion_mnist_repair_data_dir

    draw_bubble_chart.run(
        test_dir=str(data_dir),
        model_dir=str(pretrained_keras_model_dir),
        output_dir=str(data_dir),
    )

    assert (data_dir / "bubble.png").exists()
