import numpy as np
import pytest

from transport_data_analysis import peakdet


def test_peakdet_finds_maxima_and_minima():
    maxima, minima = peakdet([0, 2, 0, -2, 0], delta=1)

    np.testing.assert_array_equal(maxima, [[1, 2]])
    np.testing.assert_array_equal(minima, [[3, -2]])


def test_peakdet_returns_stable_empty_shapes():
    maxima, minima = peakdet([0, 0, 0], delta=1)

    assert maxima.shape == (0, 2)
    assert minima.shape == (0, 2)


@pytest.mark.parametrize("delta", [0, -1])
def test_peakdet_rejects_non_positive_delta(delta):
    with pytest.raises(ValueError, match="positive"):
        peakdet([0, 1], delta)
