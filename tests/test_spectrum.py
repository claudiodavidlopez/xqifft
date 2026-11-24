"""Tests of `xqifft.spectrum`."""

import numpy as np
import pytest

from xqifft.spectrum import interpolate_at, next_power2, spectrum_phase, windowed_fft


@pytest.mark.parametrize(
    ("number", "power"),
    [(1, 1), (2, 2), (3, 4), (4, 4), (5, 8), (8, 8), (9, 16), (1000, 1024)],
)
def test_next_power2(number: int, power: int) -> None:
    """Return the smallest power of 2 not less than the number."""
    assert next_power2(number) == power


def test_next_power2_rejects_non_positive() -> None:
    """Reject numbers below 1, which have no power of 2 above them."""
    with pytest.raises(ValueError, match="at least 1"):
        next_power2(0)


def test_windowed_fft_of_bin_centered_cosine() -> None:
    """Give two peaks of half the amplitude at the cosine's frequency."""
    time_step = 1e-3
    time = np.arange(64) * time_step
    frequency = 8 / (64 * time_step)
    signal = 3.0 * np.cos(2 * np.pi * frequency * time)

    spectrum, frequencies = windowed_fft(signal, time_step, np.ones(64))

    assert len(spectrum) == 64
    np.testing.assert_allclose(np.diff(frequencies), 1 / (64 * time_step))
    peaks = np.sort(frequencies[np.abs(spectrum) > 1.0])
    np.testing.assert_allclose(peaks, [-frequency, frequency])
    np.testing.assert_allclose(np.abs(spectrum).max(), 1.5)


def test_windowed_fft_pads_to_power_of_2() -> None:
    """Pad the spectrum to the next power of 2 of the padded length."""
    signal = np.ones(100)

    spectrum, frequencies = windowed_fft(signal, 0.01, np.hanning(100), padding=3.0)

    assert len(spectrum) == len(frequencies) == 512
    assert np.abs(spectrum).max() == pytest.approx(1.0)


def test_windowed_fft_rejects_padding_below_one() -> None:
    """Reject a padding factor that would truncate the signal."""
    with pytest.raises(ValueError, match="padding"):
        windowed_fft(np.ones(8), 0.1, np.ones(8), padding=0.5)


def test_windowed_fft_rejects_mismatched_window() -> None:
    """Reject a window of another length than the signal."""
    with pytest.raises(ValueError, match="same length"):
        windowed_fft(np.ones(8), 0.1, np.ones(7))


def test_spectrum_phase_wrapped_and_unwrapped() -> None:
    """Return the wrapped phase, or remove its 2 * pi jumps when asked."""
    phase = np.linspace(0.0, 4 * np.pi, 9)
    spectrum = np.exp(1j * phase)

    wrapped = spectrum_phase(spectrum)
    unwrapped = spectrum_phase(spectrum, unwrap=True)

    assert np.all(np.abs(wrapped) <= np.pi)
    np.testing.assert_allclose(unwrapped, phase, atol=1e-12)


def test_interpolate_at() -> None:
    """Interpolate linearly between neighboring values."""
    values = np.array([0.0, 10.0, 30.0])

    result = interpolate_at(np.array([0.0, 0.5, 1.25, 2.0]), values)

    np.testing.assert_allclose(result, [0.0, 5.0, 15.0, 30.0])
