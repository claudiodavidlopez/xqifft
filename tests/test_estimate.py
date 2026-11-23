"""Tests of `xqifft.estimate`."""

import numpy as np
import pytest

from xqifft.estimate import refine_phases, xqifft
from xqifft.fourier import FourierSeries

TIME_STEP = 1e-4
TIME = np.arange(1000) * TIME_STEP


def make_series(offset: float) -> FourierSeries:
    """Return a 50 Hz signal with two smaller odd harmonics and `offset`."""
    return FourierSeries(
        offset=offset,
        amplitudes=np.array([100.0, 5.0, 1.0]),
        frequencies=np.array([50.0, 150.0, 250.0]),
        phases=np.array([np.pi / 2, 0.0, np.pi / 8]),
    )


@pytest.mark.parametrize("offset", [0.0, 3.0])
@pytest.mark.parametrize("unwrap_phase", [False, True])
def test_xqifft_recovers_coefficients(offset: float, unwrap_phase: bool) -> None:
    """Estimate offset, amplitudes and frequencies of five cycles closely."""
    true = make_series(offset)

    estimate = xqifft(
        true(TIME), TIME_STEP, zero_padding=10.0, unwrap_phase=unwrap_phase
    )

    assert estimate.offset == pytest.approx(offset, abs=1e-9)
    np.testing.assert_allclose(estimate.amplitudes, true.amplitudes, rtol=1e-5)
    np.testing.assert_allclose(estimate.frequencies, true.frequencies, rtol=2e-4)
    phase_error = np.angle(np.exp(1j * (estimate.phases - true.phases)))
    np.testing.assert_allclose(phase_error, 0.0, atol=2e-2)


def test_xqifft_threshold_drops_small_harmonics() -> None:
    """Ignore harmonics below the side lobe threshold."""
    estimate = xqifft(make_series(0.0)(TIME), TIME_STEP, side_lobe_threshold=0.099)

    np.testing.assert_allclose(estimate.frequencies, [50.0], rtol=1e-3)


def test_refine_phases_reduces_error() -> None:
    """Fit only the phases, and reconstruct the signal better than before."""
    true = make_series(0.0)
    signal = true(TIME)
    estimate = xqifft(signal, TIME_STEP, zero_padding=10.0)
    long_time = np.arange(10000) * TIME_STEP

    refined = refine_phases(estimate, TIME, signal)

    np.testing.assert_array_equal(refined.amplitudes, estimate.amplitudes)
    np.testing.assert_array_equal(refined.frequencies, estimate.frequencies)
    error = np.abs(true(long_time) - estimate(long_time)).max()
    refined_error = np.abs(true(long_time) - refined(long_time)).max()
    assert refined_error < error
