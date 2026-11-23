# xqifft

[![Checks](https://github.com/claudiodavidlopez/xqifft/actions/workflows/checks.yml/badge.svg)](https://github.com/claudiodavidlopez/xqifft/actions/workflows/checks.yml)

Estimate the Fourier series of a sampled signal (DC offset, and the
amplitude, frequency and phase of each harmonic) with the XQIFFT algorithm.

The XQIFFT was proposed by K. J. Werner in
["The XQIFFT: Increasing the accuracy of quadratic interpolation of spectral peaks via exponential magnitude spectrum weighting"](https://quod.lib.umich.edu/i/icmc/bbp2372.2015.066/1/--xqifft-increasing-the-accuracy-of-quadratic-interpolation?page=root;size=150;view=pdf)
(ICMC 2015). This implementation was written for the paper
["Speeding Up AC Circuit Co-Simulations through Selective Simulator Decoupling of Predictable States"](https://doi.org/10.1109/ACCESS.2019.2907773)
(IEEE Access, 2019).

## Contents

- [How it works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
- [API](#api)
- [Examples](#examples)
- [Accuracy and limitations](#accuracy-and-limitations)
- [Development](#development)
- [Design decisions](#design-decisions)
- [Citing](#citing)
- [License](#license)

## How it works

`xqifft.estimate.xqifft` runs these steps on a uniformly sampled real signal:

1. Apply a Blackman window and compute the zero-padded FFT, scaled so that
   a cosine of amplitude A gives two peaks of magnitude A / 2.
2. Find the main lobe peaks of the magnitude spectrum, ignoring everything
   below a threshold relative to the largest peak (side lobes).
3. Refine each peak with the XQIFFT: weight the magnitudes with
   x ** p, fit a parabola through the peak and its two neighbors, and take
   its vertex as the location and height of the continuous peak.
4. Read the frequency and the phase of the spectrum at each refined
   location by linear interpolation.
5. Fold the symmetric spectrum of the real signal into a DC offset and
   positive-frequency harmonics.

The weighting makes the main lobe of the window closer to a parabola, so
quadratic interpolation becomes much more accurate than on the plain
magnitude (QIFFT).

## Installation

Requires Python 3.12 or later. The package is not on PyPI; install it from
GitHub:

```sh
pip install git+https://github.com/claudiodavidlopez/xqifft
```

or, in a uv project:

```sh
uv add git+https://github.com/claudiodavidlopez/xqifft
```

Add `@v0.2.0` (or another tag) to the URL to install a fixed release.

The runtime dependencies are NumPy (1.26 or later) and SciPy (1.11.2 or
later).

## Usage

```python
import numpy as np

from xqifft.estimate import refine_phases, xqifft

# Five cycles of a 50 Hz signal with a DC offset and two odd harmonics.
time_step = 1e-4  # s
time = np.arange(1000) * time_step
signal = (
    -3.0
    + 100.0 * np.cos(2 * np.pi * 50.0 * time + np.pi / 2)
    + 5.0 * np.cos(2 * np.pi * 150.0 * time)
    + 1.0 * np.cos(2 * np.pi * 250.0 * time + np.pi / 8)
)

series = xqifft(signal, time_step, zero_padding=10.0)

series.offset  # -3.0000000000000004
series.amplitudes  # [100.00000881   5.00000003   1.0000047 ]
series.frequencies  # [ 50.00406051 150.00207348 249.96820263], in Hz
series.phases  # [ 1.56952062e+00 -6.51542147e-04  4.02697394e-01], in rad

# Fit the phases to the samples by least squares.
refined = refine_phases(series, time, signal)

# A FourierSeries is callable: evaluate it at one time or at an array of times.
refined(0.05)
refined(np.linspace(0.0, 0.1, 5))
```

The result is a `FourierSeries`, an immutable dataclass with the fields
`offset`, `amplitudes`, `frequencies` and `phases`. Its value at time t is

```
offset + sum(amplitudes[k] * cos(2 * pi * frequencies[k] * t + phases[k]))
```

Use `dataclasses.replace(series, phases=...)` to derive a series with other
values.

## API

Everything is imported from its submodule; the top-level package exports
nothing.

| Module | Contents |
|---|---|
| `xqifft.estimate` | `xqifft(signal, time_step, zero_padding=1.0, unwrap_phase=True, side_lobe_threshold=0.0099, exponent=0.2308)`, the whole estimation; `refine_phases(series, time, signal)` |
| `xqifft.fourier` | `FourierSeries`; `series_from_peaks`, which folds spectrum peaks into a series; `wrap_phase` |
| `xqifft.spectrum` | `windowed_fft`, `spectrum_phase`, `interpolate_at`, `next_power2` |
| `xqifft.peaks` | `discrete_magnitude_peaks`, `continuous_magnitude_peaks`, and the defaults `DEFAULT_SIDE_LOBE_THRESHOLD` and `DEFAULT_EXPONENT` |

Every public function has a docstring with the units of its arguments and
results; see `help(xqifft.estimate.xqifft)`, for example.

The parameters of `xqifft`:

| Parameter | Meaning |
|---|---|
| `zero_padding` | Factor, at least 1, by which the signal is zero-padded before the FFT; the length is then rounded up to a power of 2. More padding gives a finer spectrum and better phase estimates. |
| `unwrap_phase` | Unwrap the spectrum phase before interpolating it at the peaks. |
| `side_lobe_threshold` | Fraction of the largest peak below which the spectrum is ignored. The default 0.0099 is about -40 dB, well above the side lobes of the Blackman window (about -58 dB). |
| `exponent` | Exponent p of the weighting x ** p. The best value depends on the window; the default suits the Blackman window. |

## Examples

The [examples](examples/) need matplotlib, which is in the development
dependencies:

```sh
uv run python examples/plot_pipeline.py   # plot every step of the estimation
uv run python examples/reconstruct.py     # estimate from 5 cycles, compare over 50
```

## Accuracy and limitations

- The signal must be real and uniformly sampled, and should hold a few
  cycles of its lowest frequency. Two harmonics closer than the width of a
  main lobe (about 6 bins of the unpadded spectrum for the Blackman window)
  merge into one peak.
- A harmonic smaller than `side_lobe_threshold` times the largest one is
  dropped.
- In the example above, amplitudes are within 1e-5 relative, frequencies
  within 0.004 Hz for the fundamental and 0.04 Hz for the smallest
  harmonic, and phases within about 0.01 rad. `refine_phases` improves the
  phases further.
- An error in frequency turns into a growing phase error when the series is
  evaluated far beyond the samples. With the 0.004 Hz error of the
  fundamental, its phase drifts by about 0.013 rad after 0.5 s.
- The offset and harmonics are found by assuming the peaks are symmetric
  around 0 Hz, as they are for a real signal. Noise that adds a peak on one
  side only breaks that assumption.

## Development

The project uses [uv](https://docs.astral.sh/uv/). Clone the repository, then:

```sh
uv sync                                  # create .venv with all dependencies
git config core.hooksPath .githooks      # run the checks before every push
```

The checks, also run by the pre-push hook and by GitHub Actions on Linux and
Windows with Python 3.12, 3.13 and 3.14:

```sh
uv run ruff check .          # lint, including docstrings
uv run ruff format --check . # formatting
uv run mypy                  # strict type check of src, tests and examples
uv run pytest --cov          # tests, with 100% branch coverage required
```

`uv run pytest tests/test_peaks.py` runs a single file without the coverage
gate. GitHub Actions also runs the tests with the lowest supported versions
of NumPy and SciPy.

Layout:

```
src/xqifft/     the package (typed, ships py.typed)
tests/          pytest tests, one file per module
examples/       runnable scripts with plots
```

Releases follow [Semantic Versioning](https://semver.org/); changes are
listed in [CHANGELOG.md](CHANGELOG.md). A release is a tag `vX.Y.Z` on the
default branch, matching `version` in `pyproject.toml`.

## Design decisions

- **Results are a `FourierSeries`**, not a tuple of four values. It checks
  its arrays, cannot be changed after it is created, and can be evaluated
  directly, which is what the co-simulation use case needs.
- **No re-exports from the top-level package.** Imports name the submodule,
  so it is always clear where a function lives and the package can grow
  without one large namespace.
- **Docstrings follow the Google style** with the type and unit of every
  argument and result, in plain text without Sphinx markup, since there is
  no generated documentation.
- **CI runs on GitHub Actions** in addition to the pre-push hook, because
  the repository is public and receives changes through pull requests.

## Citing

If you use this package in your research, please cite the paper that
proposed the algorithm:

> K. J. Werner, "The XQIFFT: Increasing the accuracy of quadratic
> interpolation of spectral peaks via exponential magnitude spectrum
> weighting," in Proceedings of the International Computer Music
> Conference (ICMC), 2015.

and, if relevant, the paper this implementation was written for:
<https://doi.org/10.1109/ACCESS.2019.2907773>.

## License

[MIT](LICENSE)
