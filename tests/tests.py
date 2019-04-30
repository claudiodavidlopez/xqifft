import numpy as np
import pytest

import xqifft


def test_fourier_signal_array():
    """Test that `fourier_signal()` returns the right signal.
    """
    # Define signal coefficients
    A0 = 1.3
    A1 = 2.4
    A2 = 3.2
    f1 = 50.0
    f2 = 100.0
    p1 = np.pi
    p2 = np.pi/2
    # Generate time vector
    t, _ = np.linspace(0.0, 0.04, 400, endpoint=False, retstep=True)
    # Assert t type
    assert isinstance(t, np.ndarray)
    # Generate test signal
    s0 = A0 \
       + A1*np.cos(2*np.pi*f1*t + p1)  \
       + A2*np.cos(2*np.pi*f2*t + p2)
    # Generate fourier signal from coefficients
    s1 = xqifft.fourier_signal(
        t,
        A0,
        np.array([A1, A2]),
        np.array([f1, f2]),
        np.array([p1, p2]))
    # Compare signals
    assert np.allclose(s0, s1)
    # Assert result type
    assert isinstance(s1, np.ndarray)


def test_fourier_signal_float():
    """Test that `fourier_signal()` returns the right value.
    """
    # Define signal coefficients
    A0 = 1.3
    A1 = 2.4
    A2 = 3.2
    f1 = 50.0
    f2 = 100.0
    p1 = np.pi
    p2 = np.pi/2
    # Set a time
    t = 0.021
    # Assert t type
    assert isinstance(t, float)
    # Calculate signal value for time
    s0 = A0 \
       + A1*np.cos(2*np.pi*f1*t + p1)  \
       + A2*np.cos(2*np.pi*f2*t + p2)
    # Calculate fourier signal value at time from coefficients
    s1 = xqifft.fourier_signal(
        t,
        A0,
        np.array([A1, A2]),
        np.array([f1, f2]),
        np.array([p1, p2]))
    # Compare
    assert s0 == pytest.approx(s1)
    # Assert result type
    assert isinstance(s1, float)


def test_fourier_signal_except():
    """Test that `fourier_signal()` raises the expected exceptions.
    """
    # Test RuntimeError for argument lengths
    with pytest.raises(RuntimeError):
        xqifft.fourier_signal(
            np.array([0.0, 1.0]),
            0.0,
            np.array([1.0, 0.0]),
            np.array([50.0]),
            np.array([np.pi]))

    with pytest.raises(RuntimeError):
        xqifft.fourier_signal(
            np.array([0.0, 1.0]),
            0.0,
            np.array([1.0]),
            np.array([50.0, 0.0]),
            np.array([np.pi]))

    with pytest.raises(RuntimeError):
        xqifft.fourier_signal(
            np.array([0.0, 1.0]),
            0.0,
            np.array([1.0]),
            np.array([50.0]),
            np.array([np.pi, 0.0]))


def test_fourier_signal_callable_array():
    """Test that `fourier_signal_callable()` returns the right signal.
    """
    # Define signal coefficients
    A0 = 1.3
    A1 = 2.4
    A2 = 3.2
    f1 = 50.0
    f2 = 100.0
    p1 = np.pi
    p2 = np.pi/2
    # Generate time vector
    t, _ = np.linspace(0.0, 0.04, 400, endpoint=False, retstep=True)
    # Assert t type
    assert isinstance(t, np.ndarray)
    # Generate test signal
    s0 = A0 \
       + A1*np.cos(2*np.pi*f1*t + p1)  \
       + A2*np.cos(2*np.pi*f2*t + p2)
    # Generate fourier signal callable from coefficients
    s1_callable = xqifft.fourier_signal_callable(
        A0,
        np.array([A1, A2]),
        np.array([f1, f2]),
        np.array([p1, p2]))
    # Generate fourier signal from callable
    s1 = s1_callable(t)
    # Compare signals
    assert np.allclose(s0, s1)
    # Assert result type
    assert isinstance(s1, np.ndarray)


def test_fourier_signal_callable_float():
    """Test that `fourier_signal_callable()` returns the right value.
    """
    # Define signal coefficients
    A0 = 1.3
    A1 = 2.4
    A2 = 3.2
    f1 = 50.0
    f2 = 100.0
    p1 = np.pi
    p2 = np.pi/2
    # Set a time
    t = 0.021
    # Assert t type
    assert isinstance(t, float)
    # Calculate signal value for time
    s0 = A0 \
       + A1*np.cos(2*np.pi*f1*t + p1)  \
       + A2*np.cos(2*np.pi*f2*t + p2)
    # Create fourier signal callable from coefficients
    s1_callable = xqifft.fourier_signal_callable(
        A0,
        np.array([A1, A2]),
        np.array([f1, f2]),
        np.array([p1, p2]))
    # Calculate signal value from callable
    s1 = s1_callable(t)
    # Compare
    assert s0 == pytest.approx(s1)
    # Assert result type
    assert isinstance(s1, float)


def test_next_power2():
    """Test that `next_power2()` returns right values.
    """
    assert xqifft.next_power2(1) == 2**0
    assert xqifft.next_power2(2) == 2**1
    assert xqifft.next_power2(3) == 2**2
    assert xqifft.next_power2(4) == 2**2
    assert xqifft.next_power2(5) == 2**3
    assert xqifft.next_power2(6) == 2**3
    assert xqifft.next_power2(7) == 2**3
    assert xqifft.next_power2(8) == 2**3
    assert xqifft.next_power2(9) == 2**4
    assert xqifft.next_power2(9) == 2**4
