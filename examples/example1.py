import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sg

from context import xqifft


ZERO_PADDING = 10.0
UNWRAP_PHASE = True
SIDE_LOBE_THRESHOLD = 0.0099


if __name__ == '__main__':
    # Fourier coefficients
    A_0 = 3.0
    A_n = [100.0, 5.0, 1.0]
    f_n = [50.0, 150.0, 250.0]
    p_n = [np.pi/2, 0.0, np.pi]

    # Generate time vector
    t, dt = np.linspace(0, 0.08, 8000, endpoint=False, retstep=True)
    # Generate signal
    s = xqifft.fourier_signal(t, A_0, A_n, f_n, p_n)
    # Generate window
    window = sg.blackman(s.size, sym=False)

    # Get complex spectrum
    S, S_f = xqifft.windowed_fft(s, dt, window, padding=ZERO_PADDING)
    # Get spectrum frequency resolution
    df = xqifft.frequency_resolution(S_f)
    # Get spectrum magnitude
    S_m = xqifft.spectrum_magnitude(S)
    # Get spectrum phase
    S_p = xqifft.spectrum_phase(S, unwrap=UNWRAP_PHASE)

    # Find discrete magnitude peaks
    dmag_peak_indices, dmag_peaks = xqifft.discrete_magnitude_peaks(
        S_m, threshold=SIDE_LOBE_THRESHOLD)

    # Find continuous magnitude peaks
    cmag_peak_indices, cmag_peaks = xqifft.continuous_magnitude_peaks(
        dmag_peak_indices, S_m)

    # Find continuous phases
    cmag_peak_phases = xqifft.continuous_phases(cmag_peak_indices, S_p)

    # Transform continuous indices to continuous bins
    cmag_peak_bins = xqifft.indices_to_bins(cmag_peak_indices, len(S))
    # Transform continuous bins to continuous frequencies
    cmag_peak_frequencies = xqifft.bins_to_frequencies(cmag_peak_bins, df)

    # Plot the whole process
    max_f = 1.25*max(f_n)

    f, axs = plt.subplots(6, 1)
    axs[0].plot(t, s, label='signal')
    axs[1].plot(S_f, S_m, label='spectrum magnitude')
    axs[2].plot(S_f, S_p, label='spectrum phase')
    axs[3].plot(S_f, S_m, label='spectrum magnitude')
    axs[3].plot(cmag_peak_frequencies, dmag_peaks, 'x', label='discrete peak')
    axs[4].plot(S_f, S_m, label='spectrum magnitude')
    axs[4].plot(cmag_peak_frequencies, dmag_peaks, '.', label='discrete peak')
    axs[4].plot(cmag_peak_frequencies, cmag_peaks, 'x', label='continuous peak')
    axs[5].plot(S_f, S_p, label='spectrum phase')
    axs[5].plot(cmag_peak_frequencies, cmag_peak_phases, 'x', label='peak phase')

    axs[0].set_xlabel('Time (s)')

    for ax in axs[1:]:
        ax.set_xlim(-max_f, max_f)
        ax.set_xlabel('Frequency (Hz)')

    for ax in axs:
        ax.legend()

    plt.show()
