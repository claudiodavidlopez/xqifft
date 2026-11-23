"""Estimate a signal from five cycles and compare it over fifty.

Shows how much `refine_phases` reduces the error of the reconstruction. Run
with `uv run python examples/reconstruct.py`.
"""

import matplotlib.pyplot as plt
import numpy as np

from xqifft.estimate import refine_phases, xqifft
from xqifft.fourier import FourierSeries

CAPTURED_CYCLES = 5
COMPARED_CYCLES = 50
SAMPLES_PER_CYCLE = 200
ZERO_PADDING = 10.0


def main() -> None:
    """Estimate, refine and reconstruct the signal, then plot the errors."""
    series = FourierSeries(
        offset=0.0,
        amplitudes=np.array([100.0, 5.0, 1.0]),
        frequencies=np.array([50.0, 150.0, 250.0]),
        phases=np.array([np.pi / 2, 0.0, np.pi / 8]),
    )
    time_step = 1 / (series.frequencies[0] * SAMPLES_PER_CYCLE)
    captured = np.arange(CAPTURED_CYCLES * SAMPLES_PER_CYCLE) * time_step
    signal = series(captured)

    estimate = xqifft(signal, time_step, zero_padding=ZERO_PADDING, unwrap_phase=False)
    refined = refine_phases(estimate, captured, signal)

    time = np.arange(COMPARED_CYCLES * SAMPLES_PER_CYCLE) * time_step
    original = series(time)

    _, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(time, original, label="Original")
    axes[0].plot(time, estimate(time), label="Reconstructed")
    axes[0].plot(time, refined(time), label="Reconstructed refined")
    axes[1].plot(time, np.abs(original - estimate(time)), label="Error")
    axes[1].plot(time, np.abs(original - refined(time)), label="Error refined")
    axes[-1].set_xlabel("Time (s)")

    for axis in axes:
        axis.legend()

    plt.show()


if __name__ == "__main__":
    main()
