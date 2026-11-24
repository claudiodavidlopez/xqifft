"""Estimate the Fourier coefficients of a signal with the XQIFFT algorithm.

The XQIFFT (K. J. Werner, ICMC 2015) refines the location and height of the
peaks of a discrete spectrum by quadratic interpolation on an exponentially
weighted magnitude spectrum.

Import from the submodules:

- `xqifft.estimate`: `xqifft`, the whole estimation, and `refine_phases`.
- `xqifft.fourier`: `FourierSeries`, the estimated signal model.
- `xqifft.spectrum`: the windowed FFT and spectrum helpers.
- `xqifft.peaks`: discrete and continuous peak detection.
"""
