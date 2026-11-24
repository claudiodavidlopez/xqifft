"""Estimate the Fourier series of a sampled signal."""

from dataclasses import replace

import numpy as np
import scipy.optimize as op
import scipy.signal as sg

from xqifft._typing import FloatArray
from xqifft.fourier import FourierSeries, series_from_peaks
from xqifft.peaks import (
    DEFAULT_EXPONENT,
    DEFAULT_SIDE_LOBE_THRESHOLD,
    continuous_magnitude_peaks,
    discrete_magnitude_peaks,
)
from xqifft.spectrum import interpolate_at, spectrum_phase, windowed_fft


def xqifft(
    signal: FloatArray,
    time_step: float,
    zero_padding: float = 1.0,
    unwrap_phase: bool = True,
    side_lobe_threshold: float = DEFAULT_SIDE_LOBE_THRESHOLD,
    exponent: float = DEFAULT_EXPONENT,
) -> FourierSeries:
    """Return the Fourier series of a real signal estimated with the XQIFFT.

    Applies a Blackman window, computes the zero-padded spectrum, finds its
    main lobe peaks, refines their location and magnitude with the XQIFFT and
    reads the phase of the spectrum at each refined location.

    Args:
        signal (FloatArray): Uniformly sampled real signal. It is not changed.
        time_step (float): Time between two samples of `signal`, in s.
        zero_padding (float): Zero-padding factor, at least 1; see
            `xqifft.spectrum.windowed_fft`.
        unwrap_phase (bool): Whether to unwrap the spectrum phase before
            interpolating it at the peaks.
        side_lobe_threshold (float): Fraction of the largest magnitude below
            which spectrum points are ignored as side lobes (dimensionless).
        exponent (float): Exponent of the magnitude weighting
            (dimensionless); see `xqifft.peaks.continuous_magnitude_peaks`.

    Raises:
        ValueError: If `zero_padding` is less than 1.

    Returns:
        FourierSeries: The estimated DC offset and harmonics.
    """
    window = sg.windows.blackman(len(signal), sym=False)
    spectrum, frequencies = windowed_fft(signal, time_step, window, zero_padding)
    magnitude = np.abs(spectrum)
    phase = spectrum_phase(spectrum, unwrap=unwrap_phase)
    indices = discrete_magnitude_peaks(magnitude, side_lobe_threshold)
    positions, peaks = continuous_magnitude_peaks(indices, magnitude, exponent)

    return series_from_peaks(
        peaks, interpolate_at(positions, frequencies), interpolate_at(positions, phase)
    )


def refine_phases(
    series: FourierSeries, time: FloatArray, signal: FloatArray
) -> FourierSeries:
    """Return the series with its phases fitted to the signal by least squares.

    The offset, amplitudes and frequencies are kept; only the phases are
    fitted, starting from those of `series`.

    Args:
        series (FourierSeries): Estimate to refine, for example from `xqifft`.
        time (FloatArray): Time of each sample of `signal`, in s. It is not
            changed.
        signal (FloatArray): Sampled signal. It is not changed.

    Returns:
        FourierSeries: A new series with the fitted phases, in rad.
    """

    def model(times: FloatArray, *phases: float) -> FloatArray:
        """Return the series with the given phases at each time."""
        return replace(series, phases=np.array(phases))(times)

    fitted, _ = op.curve_fit(model, time, signal, p0=series.phases)
    return replace(series, phases=fitted)
