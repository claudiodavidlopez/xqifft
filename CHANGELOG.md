# Changelog

All notable changes to this project are listed in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] - 2026-11-23

A rewrite into a typed, installable package. The API is not compatible
with 0.1.0.

### Added

- Packaging with `pyproject.toml` (build backend `uv_build`), so the package
  can be installed with pip or uv. Type information ships with `py.typed`.
- `FourierSeries`, an immutable, callable result type.
- `exponent` parameter of `xqifft`.
- Checks: ruff, mypy in strict mode, and pytest with 100% branch coverage,
  run by a pre-push hook and by GitHub Actions.

### Changed

- The code is split into the submodules `estimate`, `fourier`, `spectrum`
  and `peaks`; the top-level package exports nothing.
- `xqifft(signal, time_step, ...)` takes the time step instead of the time
  array, and returns a `FourierSeries` instead of a tuple.
- `refine_phases(series, time, signal)` takes and returns a
  `FourierSeries`.
- `fourier_signal` and `fourier_signal_callable` are replaced by calling a
  `FourierSeries`; `fourier_signal_coefficients` by `series_from_peaks`.
- `discrete_magnitude_peaks` returns only the indices of the peaks.
- `windowed_fft` names its time step `time_step` instead of `dt`.
- Invalid arguments raise `ValueError` instead of `RuntimeError`.
- Docstrings use the Google style and state units.

### Removed

- `wrap`, `time_resolution`, `frequency_resolution`, `spectrum_magnitude`
  (use `numpy.abs`), `indices_to_bins`, `bins_to_frequencies` and
  `continuous_phases` (use `interpolate_at` for both frequencies and
  phases), and the scalar `continuous_magnitude_peak` (use
  `continuous_magnitude_peaks`).
- Support for Python versions before 3.12.

### Fixed

- `xqifft` crashed with SciPy 1.13 and later, which removed
  `scipy.signal.blackman`.
- The default side lobe threshold of `xqifft` was 0.099 instead of 0.0099,
  which dropped every harmonic smaller than 10% of the largest one.
- A negative DC offset was estimated as positive.
- Phases estimated with `unwrap_phase=True` could lie far outside
  [-pi, pi); they are now wrapped.
- Peak indices at the ends of the spectrum raise `ValueError` instead of
  silently wrapping around.

## [0.1.0] - 2019

First version, as used for the paper "Speeding Up AC Circuit
Co-Simulations through Selective Simulator Decoupling of Predictable
States". Not tagged.

[Unreleased]: https://github.com/claudiodavidlopez/xqifft/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/claudiodavidlopez/xqifft/releases/tag/v0.2.0
