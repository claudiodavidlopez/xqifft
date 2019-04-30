import matplotlib.pyplot as plt
import numpy as np

from context import xqifft


CAPTURED_CICLES = 5
COMPARED_CICLES = 50
SAMPLES_PER_CICLE = 200
ZERO_PADDING = 10.0
UNWRAP_PHASE = False
SIDE_LOBE_THRESHOLD = 0.0099


if __name__ == '__main__':
    # Fourier coefficients
    A_0 = 0.0
    A_n = [100.0, 5.0, 1.0]
    f_n = [50.0, 150.0, 250.0]
    p_n = [np.pi/2, 0.0, np.pi/8]

    T = 1/f_n[0]

    # Generate time vector
    t, dt = np.linspace(
        0.0,
        CAPTURED_CICLES*T,
        CAPTURED_CICLES*SAMPLES_PER_CICLE,
        endpoint=False, retstep=True)

    # Generate signal
    s = xqifft.fourier_signal(t, A_0, A_n, f_n, p_n)

    # Estimate signal coefficients
    eA_0, eA_n, ef_n, ep_n = xqifft.xqifft(
        t,
        s,
        zero_padding=ZERO_PADDING,
        unwrap_phase=UNWRAP_PHASE,
        side_lobe_threshold=SIDE_LOBE_THRESHOLD)

    # Refine estimated phases
    rep_n = xqifft.refine_phases(t, s, eA_0, eA_n, ef_n, ep_n)

    # Re-generate time vector for COMPARED_CICLES
    t, dt = np.linspace(
        0.0,
        COMPARED_CICLES*T,
        COMPARED_CICLES*SAMPLES_PER_CICLE,
        endpoint=False,
        retstep=True)

    # Re-generate original signal for COMPARED_CICLES
    s = xqifft.fourier_signal(t, A_0, A_n, f_n, p_n)
    # Reconstruct signal from estimated coeficients
    es = xqifft.fourier_signal(t, eA_0, eA_n, ef_n, ep_n)
    # Reconstruct signal from estimated refined
    res = xqifft.fourier_signal(t, eA_0, eA_n, ef_n, rep_n)

    # Compare visually
    f, axs = plt.subplots(2, 1, sharex=True)
    axs[0].plot(t, s, label='Original')
    axs[0].plot(t, es, label='Reconstructed')
    axs[0].plot(t, res, label='Reconstructed refined')
    axs[1].plot(t, np.abs(s-es), label='Error')
    axs[1].plot(t, np.abs(s-res), label='Error refined')

    axs[-1].set_xlabel('Time (s)')

    for ax in axs:
        ax.legend()

    plt.show()
