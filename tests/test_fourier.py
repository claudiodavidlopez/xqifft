"""Tests of `xqifft.fourier`."""

import numpy as np
import pytest

from xqifft.fourier import FourierSeries, series_from_peaks


@pytest.fixture
def series() -> FourierSeries:
    """Return a series with an offset and two harmonics."""
    return FourierSeries(
        offset=1.3,
        amplitudes=np.array([2.4, 3.2]),
        frequencies=np.array([50.0, 100.0]),
        phases=np.array([np.pi, np.pi / 2]),
    )


def expected_value(time: float | np.ndarray) -> float | np.ndarray:
    """Return the value of the `series` fixture written out by hand."""
    return (
        1.3
        + 2.4 * np.cos(2 * np.pi * 50.0 * time + np.pi)
        + 3.2 * np.cos(2 * np.pi * 100.0 * time + np.pi / 2)
    )


def test_series_at_array_of_times(series: FourierSeries) -> None:
    """Evaluate the series at every time of an array."""
    time = np.linspace(0.0, 0.04, 400, endpoint=False)

    values = series(time)

    assert isinstance(values, np.ndarray)
    np.testing.assert_allclose(values, expected_value(time))


def test_series_at_scalar_time(series: FourierSeries) -> None:
    """Evaluate the series at a single time as a float."""
    value = series(0.021)

    assert isinstance(value, float)
    assert value == pytest.approx(expected_value(0.021))


def test_series_without_harmonics_is_constant() -> None:
    """Return only the offset when the series has no harmonics."""
    empty = np.array([])
    series = FourierSeries(-2.0, empty, empty, empty)

    np.testing.assert_array_equal(series(np.array([0.0, 1.0])), [-2.0, -2.0])


@pytest.mark.parametrize(
    ("amplitudes", "frequencies", "phases"),
    [
        ([1.0, 0.0], [50.0], [np.pi]),
        ([1.0], [50.0, 0.0], [np.pi]),
        ([1.0], [50.0], [np.pi, 0.0]),
        ([[1.0]], [[50.0]], [[np.pi]]),
    ],
)
def test_series_rejects_mismatched_arrays(
    amplitudes: list[float], frequencies: list[float], phases: list[float]
) -> None:
    """Reject arrays of different lengths or with more than one dimension."""
    with pytest.raises(ValueError, match="1-D and equal size"):
        FourierSeries(
            0.0, np.array(amplitudes), np.array(frequencies), np.array(phases)
        )


def test_series_copies_and_freezes_arrays() -> None:
    """Keep the series unchanged when the caller changes its own arrays."""
    amplitudes = np.array([1.0])
    series = FourierSeries(0, amplitudes, np.array([50.0]), np.array([0.0]))
    amplitudes[0] = 5.0

    assert series.amplitudes[0] == 1.0
    assert isinstance(series.offset, float)

    with pytest.raises(ValueError, match="read-only"):
        series.amplitudes[0] = 5.0


def test_series_from_symmetric_peaks_without_offset() -> None:
    """Build the harmonics from the positive half of an even number of peaks."""
    series = series_from_peaks(
        magnitudes=np.array([0.5, 2.0, 2.0, 0.5]),
        frequencies=np.array([-150.0, -50.0, 50.0, 150.0]),
        phases=np.array([-0.2, -0.1, 0.1, 0.2]),
    )

    assert series.offset == 0.0
    np.testing.assert_array_equal(series.amplitudes, [4.0, 1.0])
    np.testing.assert_array_equal(series.frequencies, [50.0, 150.0])
    np.testing.assert_allclose(series.phases, [0.1, 0.2])


def test_series_from_peaks_with_offset() -> None:
    """Take the offset from the middle peak."""
    series = series_from_peaks(
        magnitudes=np.array([2.0, 3.0, 2.0]),
        frequencies=np.array([-50.0, 0.0, 50.0]),
        phases=np.array([-0.1, 0.0, 0.1]),
    )

    assert series.offset == pytest.approx(3.0)
    np.testing.assert_array_equal(series.amplitudes, [4.0])
    np.testing.assert_allclose(series.phases, [0.1])
