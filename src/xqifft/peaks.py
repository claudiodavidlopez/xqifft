"""Find the peaks of a magnitude spectrum and refine them with the XQIFFT."""

import numpy as np
import scipy.signal as sg

from xqifft._typing import FloatArray, IndexArray

DEFAULT_SIDE_LOBE_THRESHOLD = 0.0099
"""Default main lobe threshold: 0.99% of the largest peak, about -40 dB."""

DEFAULT_EXPONENT = 0.2308
"""Default exponent of the magnitude weighting, used with the Blackman window."""


def discrete_magnitude_peaks(
    magnitude: FloatArray, threshold: float = DEFAULT_SIDE_LOBE_THRESHOLD
) -> IndexArray:
    """Return the indices of the main lobe peaks of a magnitude spectrum.

    Points below `threshold` times the largest magnitude are taken to belong
    to side lobes and are ignored. Every strict local maximum among the
    remaining points is a main lobe peak.

    Args:
        magnitude (FloatArray): Magnitude of a discrete spectrum. It is not
            changed.
        threshold (float): Fraction of the largest magnitude that separates
            main lobes from side lobes (dimensionless).

    Returns:
        IndexArray: Index in `magnitude` of each main lobe peak, in increasing
        order. Never the first or the last index.
    """
    main_lobes = np.where(magnitude < threshold * np.max(magnitude), 0.0, magnitude)
    indices: IndexArray = sg.argrelmax(main_lobes)[0]
    return indices


def continuous_magnitude_peaks(
    indices: IndexArray, magnitude: FloatArray, exponent: float = DEFAULT_EXPONENT
) -> tuple[FloatArray, FloatArray]:
    """Return the location and magnitude of each peak of the continuous spectrum.

    Fits a parabola through each discrete peak and its two neighbors, after
    weighting the magnitudes with x ** `exponent`, and takes the vertex of
    the parabola. The weighting makes the main lobe closer to a parabola,
    which reduces the error of the estimate (K. J. Werner, "The XQIFFT:
    Increasing the accuracy of quadratic interpolation of spectral peaks via
    exponential magnitude spectrum weighting", ICMC 2015).

    Args:
        indices (IndexArray): Index in `magnitude` of each discrete peak, as
            returned by `discrete_magnitude_peaks`. It is not changed.
        magnitude (FloatArray): Magnitude of the discrete spectrum, in the
            unit of the signal. It is not changed.
        exponent (float): Exponent of the weighting (dimensionless). The best
            value depends on the window; the default suits the Blackman
            window.

    Raises:
        ValueError: If an index is the first or last of `magnitude`, so that
            it lacks a neighbor.

    Returns:
        tuple[FloatArray, FloatArray]: The fractional index of each peak,
        between 0 and `len(magnitude) - 1`, and its estimated magnitude, in
        the unit of the signal.
    """
    if np.any(indices < 1) or np.any(indices > len(magnitude) - 2):
        message = "every peak index needs a neighbor on each side"
        raise ValueError(message)

    weighted = magnitude**exponent
    alpha = weighted[indices - 1]
    beta = weighted[indices]
    gamma = weighted[indices + 1]
    curvature = alpha - 2 * beta + gamma

    positions: FloatArray = indices + (alpha - gamma) / (2 * curvature)
    peaks: FloatArray = (beta - (alpha - gamma) ** 2 / (8 * curvature)) ** (
        1 / exponent
    )

    return positions, peaks
