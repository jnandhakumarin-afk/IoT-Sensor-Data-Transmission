from __future__ import annotations

import numpy as np


def snr_linear_from_db(snr_db: float) -> float:
    return 10 ** (snr_db / 10.0)


def awgn_channel(signal: np.ndarray, snr_db: float) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    signal_power = float(np.mean(np.square(signal)))
    if signal_power == 0:
        return signal.copy(), np.zeros_like(signal), 0.0, 0.0, 0.0

    snr_value = snr_linear_from_db(snr_db)
    noise_power = signal_power / snr_value
    noise_sigma = float(np.sqrt(noise_power))
    noise = noise_sigma * np.random.randn(len(signal))
    noisy_signal = signal + noise

    return noisy_signal, noise, signal_power, noise_power, noise_sigma


def actual_snr_db(signal: np.ndarray, noise: np.ndarray) -> float:
    signal_power = float(np.mean(np.square(signal)))
    noise_power = float(np.mean(np.square(noise)))
    if noise_power == 0:
        return 1000.0
    return 10.0 * np.log10(signal_power / noise_power)
