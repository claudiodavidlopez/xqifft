"""Tests of `xqifft.peaks`."""

import numpy as np
import pytest

from xqifft.peaks import continuous_magnitude_peaks, discrete_magnitude_peaks


def test_discrete_peaks_skip_side_lobes() -> None:
    """Find the main lobe peaks and ignore local maxima under the threshold."""
    magnitude = np.array([0.0, 0.001, 0.0, 1.0, 2.0, 1.0, 0.0, 0.5, 0.0, 0.002, 0.0])

    indices = discrete_magnitude_peaks(magnitude, threshold=0.01)

    np.testing.assert_array_equal(indices, [4, 7])


def test_discrete_peaks_of_silent_spectrum() -> None:
    """Find no peaks in a spectrum without any."""
    assert len(discrete_magnitude_peaks(np.zeros(16))) == 0


@pytest.mark.parametrize("exponent", [0.2308, 0.5, 1.0])
def test_continuous_peak_of_weighted_parabola_is_exact(exponent: float) -> None:
    """Recover the vertex exactly when the weighted magnitude is a parabola."""
    vertex, height = 5.3, 7.0
    index = np.arange(11, dtype=np.float64)
    weighted = height**exponent - 0.05 * (index - vertex) ** 2
    magnitude = weighted ** (1 / exponent)

    positions, peaks = continuous_magnitude_peaks(np.array([5]), magnitude, exponent)

    np.testing.assert_allclose(positions, [vertex])
    np.testing.assert_allclose(peaks, [height])


@pytest.mark.parametrize("index", [0, 4])
def test_continuous_peak_needs_neighbors(index: int) -> None:
    """Reject a peak at either end of the spectrum."""
    with pytest.raises(ValueError, match="neighbor"):
        continuous_magnitude_peaks(np.array([index]), np.ones(5))
