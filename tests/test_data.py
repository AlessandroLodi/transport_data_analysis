import numpy as np
import pytest

from transport_data_analysis import Data


def make_data():
    return Data(
        {"voltage": np.arange(5.0), "current": np.arange(5.0) ** 2},
        axes=["voltage", "current"],
    )


def test_column_and_mask_indexing_are_independent():
    data = make_data()

    selected = data[data["voltage"] >= 2]
    selected["current"] = np.zeros(3)

    np.testing.assert_array_equal(selected["voltage"].values, [2, 3, 4])
    np.testing.assert_array_equal(data["current"].values, [0, 1, 4, 9, 16])
    assert len(data) == 5


def test_data_validates_columns_and_assignment_shapes():
    with pytest.raises(ValueError, match="same shape"):
        Data({"x": np.arange(2), "y": np.arange(3)})

    data = make_data()
    with pytest.raises(ValueError, match="existing data shape"):
        data["other"] = np.arange(2)


def test_delimited_file_round_trip(tmp_path):
    source = tmp_path / "measurement.tsv"
    source.write_text("0\t1\n2\t3\n", encoding="utf-8")

    data = Data.load_from_file(source, axes=("voltage", "current"))

    np.testing.assert_array_equal(data["voltage"].values, [0, 2])
    np.testing.assert_array_equal(data["current"].values, [1, 3])

    destination = tmp_path / "saved.tsv"
    data.save(destination)
    assert destination.exists()


def test_gaussian_smoothing_returns_the_transformed_object():
    data = make_data()["current"]
    result = data.smooth(method="gaussian1")

    assert result is data
    assert result.values.shape == (5,)
