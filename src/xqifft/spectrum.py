"""Compute the windowed spectrum of a signal and read values from it."""

import numpy as np

from xqifft._typing import ComplexArray, FloatArray


def next_power2(number: int) -> int:
    """Return the smallest power of 2 that is greater than or equal to `number`.

    Args:
        number (int): A positive integer.

    Raises:
        ValueError: If `number` is less than 1.

    Returns:
        int: The power of 2.
    """
    if number < 1:
        message = "number must be at least 1"
        raise ValueError(message)

    return 1 << (number - 1).bit_length()


def windowed_fft(
    signal: FloatArray, time_step: float, window: FloatArray, padding: float = 1.0
) -> tuple[ComplexArray, FloatArray]:
    """Return the spectrum of a windowed, zero-padded signal.

    The spectrum is scaled by the sum of the window, so that a cosine of
    amplitude A that falls on a bin gives two peaks of magnitude A / 2. It is
    shifted so that the frequencies go from negative to positive.

    Args:
        signal (FloatArray): Uniformly sampled signal. It is not changed.
        time_step (float): Time between two samples of `signal`, in s.
        window (FloatArray): Window applied to `signal`, of the same length.
            It is not changed.
        padding (float): Zero-padding factor, at least 1. The spectrum length
            is the smallest power of 2 not less than `padding * len(signal)`.

    Raises:
        ValueError: If `padding` is less than 1, or if `signal` and `window`
            differ in length.

    Returns:
        tuple[ComplexArray, FloatArray]: The complex spectrum, in the unit of
        the signal, and the frequency of each of its points, in Hz.
    """
    if padding < 1.0:
        message = "padding must be at least 1.0"
        raise ValueError(message)

    if len(signal) != len(window):
        message = "signal and window must have the same length"
        raise ValueError(message)

    spectrum_length = next_power2(int(padding * len(signal)))
    spectrum = np.fft.fft(window * signal, n=spectrum_length) / np.sum(window)
    frequencies = np.fft.fftfreq(spectrum_length, time_step)

    return np.fft.fftshift(spectrum), np.fft.fftshift(frequencies)


def spectrum_phase(spectrum: ComplexArray, unwrap: bool = False) -> FloatArray:
    """Return the phase of a complex spectrum.

    Args:
        spectrum (ComplexArray): Complex spectrum. It is not changed.
        unwrap (bool): Whether to remove the jumps of 2 * pi between
            neighboring points, which makes the phase smooth for
            interpolation.

    Returns:
        FloatArray: Phase of each point of `spectrum`, in rad.
    """
    phase: FloatArray = np.angle(spectrum)

    if unwrap:
        unwrapped: FloatArray = np.unwrap(phase)
        return unwrapped

    return phase


def interpolate_at(positions: FloatArray, values: FloatArray) -> FloatArray:
    """Return `values` linearly interpolated at fractional array positions.

    Reads the frequency or the phase of a spectrum at the continuous peak
    locations returned by `xqifft.peaks.continuous_magnitude_peaks`.

    Args:
        positions (FloatArray): Fractional indices into `values`, between 0
            and `len(values) - 1`. It is not changed.
        values (FloatArray): Values at the integer indices. It is not changed.

    Returns:
        FloatArray: The interpolated value at each position, in the unit of
        `values`.
    """
    interpolated: FloatArray = np.interp(positions, np.arange(len(values)), values)
    return interpolated
