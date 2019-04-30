import functools
import numpy as np
import scipy.interpolate as ip
import scipy.optimize as op
import scipy.signal as sg


def wrap(phase):
    return (phase + np.pi) % (2*np.pi) - np.pi


def time_resolution(time):
    return time[1] - time[0]


def frequency_resolution(frequency):
    return frequency[1] - frequency[0]


def fourier_signal(time, offset, amplitudes, frequencies, phases):
    """Generate a signal from fourier coeficients.

    Parameters
    ----------
    time : ndarray or float
        time vector
    offset : float
        DC offset
    amplitudes : array_like
        amplitude of each harmonic
    frequencies : array_like
        frequency of each harmonic in Hz
    phases : array_like
        phase of each harmonic in rad

    Returns
    -------
    ndarray or float
        signal values at each time in `time`

    Raises
    ------
    RuntimeError
        when `amplitures`, `frequencies` and `phases` are of different
        lenghts

    """
    # Check argument lengths
    if len(amplitudes) != len(frequencies):
        raise RuntimeError('Mismatching argument lengths.')

    if len(amplitudes) != len(phases):
        raise RuntimeError('Mismatching argument lengths.')

    # Generate offset signal depending on type
    if isinstance(time, np.ndarray):
        signal = np.zeros(time.size) + offset
    else:
        signal = offset

    # Add harmonics to the offset
    for amp, frq, pha in zip(amplitudes, frequencies, phases):
        signal += amp*np.cos(2*np.pi*frq*time + pha)

    return signal


def fourier_signal_callable(offset, amplitudes, frequencies, phases):
    """Create a callable signal from fourier coeficients as function of time.

    Parameters
    ----------
    offset : float
        DC offset.
    amplitudes : array_like
        amplitude of each harmonic.
    frequencies : array_like
        frequency of each harmonic in Hz.
    phases : array_like
        phase of each harmonic in rad.

    Returns
    -------
    callable
        fourier signal as a function of time `f(t)`.

    """
    return functools.partial(
        fourier_signal,
        offset=offset,
        amplitudes=amplitudes,
        frequencies=frequencies,
        phases=phases)


def next_power2(integer):
    """Calculate smallest power of 2 larger than `number`.

    Parameters
    ----------
    integer : int
        any integer

    Returns
    -------
    int
        smallest power of 2 larger than `x`

    """
    return 1<<(integer-1).bit_length()


def windowed_fft(signal, dt, window, padding=1.0):
    """FFT of a windowed signal.

    Parameters
    ----------
    signal : array_like
        Input signal to the FFT.
    dt : float
        Time resolution of `signal`.
    window : array_like
        Window function applied to `signal`.
    padding : float
        Zero-padding factor higher or equal to 1.0. A factor of 1.0
        results in padding to the smallest power of 2 larger than
        `len(signal)`.

    Returns
    -------
    ndarray
        Complex spectrum.
    ndarray
        Frequency of each point in the spectrum.

    Raises
    ------
    ValueError
        If `padding < 1.0`.

    """
    if padding < 1.0:
        raise ValueError('Padding must be higher or equal to 1.0.')

    # Find a power of 2 spectrum length according to desired padding
    spectrum_length = next_power2(int(padding*signal.size))

    # Calculate spectrum
    spectrum = np.fft.fft(window*signal, n=spectrum_length)
    spectrum = spectrum/np.sum(window)
    spectrum = np.fft.fftshift(spectrum)

    # Calculate spectrum frequencies
    frequencies = np.fft.fftfreq(spectrum_length, dt)
    frequencies = np.fft.fftshift(frequencies)

    return spectrum, frequencies


def spectrum_magnitude(spectrum):
    """Calculate magnitude of a complex discrete specturm.

    Parameters
    ----------
    spectrum : array_like
        A complex, discrete spectrum.

    Returns
    -------
    ndarray
        The magnitude of `spectrum`.

    """
    return np.abs(spectrum)


def spectrum_phase(spectrum, unwrap=False):
    """Calculate phase of a complex discrete specturm.

    Parameters
    ----------
    spectrum : array_like
        A complex, discrete spectrum.

    Returns
    -------
    ndarray
        The phase of `spectrum`.

    """
    phase = np.angle(spectrum)

    if unwrap:
        return np.unwrap(phase)
    else:
        return phase


def discrete_magnitude_peaks(magnitude, threshold=0.0099):
    """Find all main lobe peaks in a discrete spectrum magnitude.

    Parameters
    ----------
    magnitude : array_like
        Magnitude of a discrete spectrum.
    threshold : float
        The threshold that separates between main and side lobes, in percent
        of the largest magnitude peak.

    Returns
    -------
    ndarray
        Index of each lobe peak in `magnitude` array.

    """
    # Find side lobes based on their size wrt largest peak
    max_magnitude = np.max(magnitude)
    side_lobe_indices = np.where(magnitude<max_magnitude*threshold)[0]

    # Remove side lobes from magnitude
    tmp_magnitude = np.copy(magnitude)
    tmp_magnitude[side_lobe_indices] = 0

    # Find the array index of each main lobe peak
    magnitude_peak_indices = sg.argrelmax(tmp_magnitude)[0]
    # Find the main lobe peaks
    magnitude_peaks = magnitude[magnitude_peak_indices]

    return magnitude_peak_indices, magnitude_peaks


def continuous_magnitude_peak(disc_peak_index, disc_magnitude, exponent=0.2308):
    """Estimate the value and location of a continuous spectrum peak.

    Applies quadratic interpolation to the discrete peak and its two neighbors
    to estimate the location and value of the continuous peak. Weighs the
    peaks using an exponential function to increase the accuracy of
    the estimate.

    Based on the paper The XQIFFT: Increasing the Accuracy of Quadratic
    Interpolation of Spectral Peaks via Exponeltial Magnitude Spectrum
    Weighting by Kurt James Werner.

    Parameters
    ----------
    disc_peak_index : integer
        Index in the `disc_magnitude` array where the magnitude peak
        is located.
    disc_magnitude : array_like
        Magnitude of the discrete spectrum with the peak of interest.
    exponent : float
        Exponent of the weighting function.

    Returns
    -------
    float
        A value between `0` and `len(disc_magnitude)-1` indicating the
        location of the peak in the continuous spectrum.
    float
        Estimated peak magnitude in the continuous spectrum.

    """
    # Weighing function and its inverse
    omega = lambda x: x**exponent
    inv_omega = lambda x: x**(1/exponent)

    # Weigh alpha, beta and gama
    alpha = omega(disc_magnitude[disc_peak_index-1])
    beta = omega(disc_magnitude[disc_peak_index])
    gamma = omega(disc_magnitude[disc_peak_index+1])

    # Estimate continuous peak index (peak location)
    cont_peak_index = (disc_peak_index
                       + (1/2)*(alpha - gamma)/(alpha - 2*beta + gamma))
    # Estimate continuous peak magnitude
    cont_peak = beta - (1/8)*(alpha - gamma)**2/(alpha - 2*beta + gamma)
    # Unweigh peak magnitude
    cont_peak = inv_omega(cont_peak)

    return cont_peak_index, cont_peak


def continuous_magnitude_peaks(disc_peak_indices, disc_magnitude, exponent=0.2308):
    """Estimate the value and location of each continuous spectrum peak.

    Parameters
    ----------
    disc_peak_indices : array_like
        Indices in the `disc_magnitude` array where the magnitude peaks are
        located.
    disc_magnitude : array_like
        Magnitude of the discrete spectrum with the peaks of interest.
    exponent : float
        Exponent of the weighting function for quadratic interpolation.

    Returns
    -------
    ndarray
        An array of values between `0` and `len(disc_magnitude)-1` indicating
        the location of each peak in the continuous spectrum.
    ndarray
        Estimated magnitude of each peak in the continuous spectrum.

    """
    cont_peak_indices = []
    cont_peaks = []

    for disc_peak_index in disc_peak_indices:
        cont_peak_index, cont_peak = continuous_magnitude_peak(
            disc_peak_index, disc_magnitude, exponent)

        cont_peak_indices.append(cont_peak_index)
        cont_peaks.append(cont_peak)

    cont_peak_indices = np.array(cont_peak_indices)
    cont_peaks = np.array(cont_peaks)

    return cont_peak_indices, cont_peaks


def continuous_phases(cont_indices, disc_phase):
    """Estimate phase at given indices from discrete spectrum phase.

    Interpolates `disc_phase` and finds its value at each location in
    `cont_indices`.

    Parameters
    ----------
    cont_indices : array_like
        An array of values between `0` and `len(disc_magnitude)-1` indicating
        where `disc_phase` must be interpolated.
    disc_phase : array_like
        Phase of a discrete spectrum.

    Returns
    -------
    array_like
        Phase value at each index in `cont_indices`.

    """
    disc_indices = np.arange(0, len(disc_phase))
    phase_interpolator = ip.interp1d(disc_indices, disc_phase)
    cont_phases = phase_interpolator(cont_indices)

    return cont_phases


def indices_to_bins(indices, len_spectrum):
    """Transform spectrum array indices to bin numbers.

    Parameters
    ----------
    indices : array_like or number
        Indices of the elements of a spectrum array.
    len_spectrum : integer
        Length of the discrete spectrum array.

    Returns
    -------
    ndarray
        Bin numbers corresponding to `indices`.

    """
    indices = np.array(indices)

    if len_spectrum % 2 == 0:
        bins = indices - len_spectrum/2
    else:
        bins = indices - (len_spectrum - 1)/2

    return bins


def bins_to_frequencies(bins, df):
    """Transform spectrum bin numbers (normalized frequencies) to frequencies.

    Parameters
    ----------
    bins : array_like or number
        Bin numbers.
    df : float
        Frequency resolution of the spectrum.

    Returns
    -------
    array_like or number
        Frequencies corresponding to `bins`.

    """
    return bins*df


def fourier_signal_coefficients(peak_magnitudes, peak_frequencies, peak_phases):
    """Real-frequency spectrum peaks to Fourier signal coefficients.

    Transforms the spectrum peaks, their frequencies and associated phases to
    a possitive-frequency set of coefficients that can be used as arguments
    to `fourier_signal()` to generate a discrete signal.

    Parameters
    ----------
    peak_magnitudes : array_like
        The magnitude of each peak in a spectrum.
    peak_frequencies : array_like
        The frequency of each magnitude peak in a spectrum, in the -inf to +inf
        range.
    peak_phases : array_like
        The phase associated with each magnitude peak in a spectrum.

    Returns
    -------
    float
        The DC offset of the Fourier signal.
    array_like
        The amplitude of each harmonic in the Fourier signal.
    array_like
        The frequency of each harmonic in the Fourier signal, in the 0 to +inf
        range.
    array_like
        The phase of each harmonic in the Fourier signal.

    """
    num_peaks = len(peak_magnitudes)

    if num_peaks % 2 == 0:
        index = int(num_peaks/2)
        offset = 0.0
    else:
        index = int(np.ceil(num_peaks/2))
        offset = peak_magnitudes[index-1]

    amplitudes = 2*peak_magnitudes[index:]
    frequencies = peak_frequencies[index:]
    phases = peak_phases[index:]

    return offset, amplitudes, frequencies, phases


def xqifft(time, signal, zero_padding=1, unwrap_phase=True,
           side_lobe_threshold=0.099):

    # Get signal time resolution
    dt = time_resolution(time)

    # Generate window function
    window = sg.blackman(len(signal), sym=False)

    # Get complex spectrum
    dspectrum, dspectrum_frequency = windowed_fft(
        signal, dt, window, padding=zero_padding)
    # Get spectrum frequency resolution
    df = frequency_resolution(dspectrum_frequency)
    # Get spectrum magnitude
    dspectrum_magnitude = spectrum_magnitude(dspectrum)
    # Get spectrum phase
    dspectrum_phase = spectrum_phase(dspectrum, unwrap=unwrap_phase)

    # Find discrete magnitude peaks
    dmag_peak_indices, dmag_peaks = discrete_magnitude_peaks(
        dspectrum_magnitude, threshold=side_lobe_threshold)

    # Find continuous magnitude peaks
    cmag_peak_indices, cmag_peaks = continuous_magnitude_peaks(
        dmag_peak_indices, dspectrum_magnitude)
    # Find continuous phases
    cmag_peak_phases = continuous_phases(
        cmag_peak_indices, dspectrum_phase)
    # Transform continuous indices to continuous bins
    cmag_peak_bins = indices_to_bins(cmag_peak_indices, len(dspectrum))
    # Transform continuous bins to continuous frequencies
    cmag_peak_frequencies = bins_to_frequencies(cmag_peak_bins, df)

    # Transform magnitude peaks, their frequencies and associated phases to
    # Fourier signal coefficients
    offset, amplitudes, frequencies, phases = fourier_signal_coefficients(
        cmag_peaks, cmag_peak_frequencies, cmag_peak_phases)

    return offset, amplitudes, frequencies, phases


def refine_phases(time, signal, offset, amplitudes, frequencies, phases):
    curve = lambda t, *p: fourier_signal(t, offset, amplitudes, frequencies, p)
    refined_phases, _ = op.curve_fit(curve, time, signal, p0=phases)
    return refined_phases
