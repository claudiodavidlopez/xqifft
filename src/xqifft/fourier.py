"""Fourier series: evaluate them and build them from spectrum peaks."""

from dataclasses import dataclass
from typing import overload

import numpy as np

from xqifft._typing import FloatArray


@dataclass(frozen=True, eq=False)
class FourierSeries:
    """Sum of a DC offset and cosine harmonics, as a function of time.

    The value at time t is the sum of `offset` and, for every harmonic k,
    amplitudes[k] * cos(2 * pi * frequencies[k] * t + phases[k]).

    The arrays are copied on construction and made read-only, so a series
    never changes after it is created. Use `dataclasses.replace` to derive a
    series with other values.

    Attributes:
        offset (float): DC offset, in the unit of the signal.
        amplitudes (FloatArray): Amplitude of each harmonic, in the unit of
            the signal.
        frequencies (FloatArray): Frequency of each harmonic, in Hz.
        phases (FloatArray): Phase of each harmonic at t = 0, in rad.
    """

    offset: float
    amplitudes: FloatArray
    frequencies: FloatArray
    phases: FloatArray

    def __post_init__(self) -> None:
        """Copy the arrays, make them read-only and check their shapes.

        Raises:
            ValueError: If `amplitudes`, `frequencies` and `phases` are not
                one-dimensional arrays of the same length.
        """
        arrays = {}

        for name in ("amplitudes", "frequencies", "phases"):
            array = np.array(getattr(self, name), dtype=np.float64)
            array.setflags(write=False)
            arrays[name] = array

        shapes = {array.shape for array in arrays.values()}

        if len(shapes) != 1 or len(next(iter(shapes))) != 1:
            message = "amplitudes, frequencies and phases must be 1-D and equal size"
            raise ValueError(message)

        object.__setattr__(self, "offset", float(self.offset))

        for name, array in arrays.items():
            object.__setattr__(self, name, array)

    @overload
    def __call__(self, time: float) -> float: ...

    @overload
    def __call__(self, time: FloatArray) -> FloatArray: ...

    def __call__(self, time: float | FloatArray) -> float | FloatArray:
        """Return the value of the series at each time.

        Args:
            time (float | FloatArray): Time or array of times, in s. It is not
                changed.

        Returns:
            float | FloatArray: Value at each time, in the unit of the signal.
            A float for a scalar time, otherwise an array shaped like `time`.
        """
        times = np.asarray(time, dtype=np.float64)
        arguments = 2 * np.pi * np.multiply.outer(times, self.frequencies)
        harmonics: FloatArray = np.cos(arguments + self.phases) @ self.amplitudes
        values = self.offset + harmonics

        if isinstance(time, np.ndarray):
            return values

        return float(values)


def series_from_peaks(
    magnitudes: FloatArray, frequencies: FloatArray, phases: FloatArray
) -> FourierSeries:
    """Return the Fourier series of a real signal from its spectrum peaks.

    The spectrum of a real signal is symmetric: every harmonic appears as a
    pair of peaks at -f and +f with half its amplitude each, and a DC offset
    as a single peak at 0 Hz. With the peaks sorted by frequency, an odd
    number of peaks therefore has the DC peak in the middle, and an even
    number has none.

    Args:
        magnitudes (FloatArray): Magnitude of each peak, sorted by frequency,
            in the unit of the signal.
        frequencies (FloatArray): Frequency of each peak, sorted, in Hz, from
            negative to positive.
        phases (FloatArray): Phase of the spectrum at each peak, in rad.

    Returns:
        FourierSeries: The DC offset and the positive-frequency harmonics.
    """
    count = len(magnitudes)
    first_harmonic = (count + 1) // 2

    if count % 2 == 0:
        offset = 0.0
    else:
        middle = count // 2
        offset = float(magnitudes[middle])

    return FourierSeries(
        offset=offset,
        amplitudes=2 * magnitudes[first_harmonic:],
        frequencies=frequencies[first_harmonic:],
        phases=phases[first_harmonic:],
    )
