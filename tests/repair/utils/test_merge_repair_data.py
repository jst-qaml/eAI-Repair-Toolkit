import h5py

from repair.utils import merge_repair_data


def test_merge_repair_data(targeted_data_dir):
    data_dir = targeted_data_dir
    output_dir = data_dir / "outputs"
    output_dir.mkdir()

    merge_repair_data.run(
        input_dir1=str(data_dir / "negative" / "0"),
        input_dir2=str(data_dir / "negative" / "1"),
        output_dir=output_dir,
    )

    assert (output_dir / "repair.h5").exists()

    with h5py.File(data_dir / "negative" / "0" / "repair.h5") as hf1:
        hf1_shapes = hf1["images"].shape, hf1["labels"].shape

    with h5py.File(data_dir / "negative" / "1" / "repair.h5") as hf2:
        hf2_shapes = hf2["images"].shape, hf2["labels"].shape

    with h5py.File(output_dir / "repair.h5") as hf:
        hf_shapes = hf["images"].shape, hf["labels"].shape

        assert hf_shapes[0][0] == hf1_shapes[0][0] + hf2_shapes[0][0]
        assert hf_shapes[1][0] == hf1_shapes[1][0] + hf2_shapes[1][0]

        assert hf_shapes[0][1:] == hf1_shapes[0][1:]
        assert hf_shapes[1][1:] == hf1_shapes[1][1:]
