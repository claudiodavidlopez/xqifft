"""Plot each step of the XQIFFT on a signal with an offset and three harmonics.

Run with `uv run python examples/plot_pipeline.py`.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sg

from xqifft.fourier import FourierSeries
from xqifft.peaks import continuous_magnitude_peaks, discrete_magnitude_peaks
from xqifft.spectrum import interpolate_at, spectrum_phase, windowed_fft

ZERO_PADDING = 10.0
SAMPLES = 8000
DURATION = 0.08
"""Length of the signal, in s."""


def main() -> None:
    """Estimate the peaks of the spectrum and plot every intermediate result."""
    series = FourierSeries(
        offset=3.0,
        amplitudes=np.array([100.0, 5.0, 1.0]),
        frequencies=np.array([50.0, 150.0, 250.0]),
        phases=np.array([np.pi / 2, 0.0, np.pi]),
    )
    time_step = DURATION / SAMPLES
    time = np.arange(SAMPLES) * time_step
    signal = series(time)
    window = sg.windows.blackman(SAMPLES, sym=False)

    spectrum, frequencies = windowed_fft(signal, time_step, window, ZERO_PADDING)
    magnitude = np.abs(spectrum)
    phase = spectrum_phase(spectrum, unwrap=True)
    indices = discrete_magnitude_peaks(magnitude)
    positions, peaks = continuous_magnitude_peaks(indices, magnitude)
    peak_frequencies = interpolate_at(positions, frequencies)
    peak_phases = interpolate_at(positions, phase)

    _, axes = plt.subplots(5, 1, figsize=(8, 10))
    axes[0].plot(time, signal, label="signal")
    axes[0].set_xlabel("Time (s)")
    axes[1].plot(frequencies, magnitude, label="spectrum magnitude")
    axes[2].plot(frequencies, phase, label="spectrum phase")
    axes[3].plot(frequencies, magnitude, label="spectrum magnitude")
    axes[3].plot(frequencies[indices], magnitude[indices], ".", label="discrete peak")
    axes[3].plot(peak_frequencies, peaks, "x", label="continuous peak")
    axes[4].plot(frequencies, phase, label="spectrum phase")
    axes[4].plot(peak_frequencies, peak_phases, "x", label="peak phase")
    max_frequency = 1.25 * series.frequencies.max()

    for axis in axes[1:]:
        axis.set_xlim(-max_frequency, max_frequency)
        axis.set_xlabel("Frequency (Hz)")

    for axis in axes:
        axis.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
