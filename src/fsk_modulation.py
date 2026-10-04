from __future__ import annotations

import numpy as np


def generate_bfsk_signal(bitstream: str, f0: float, f1: float, bit_rate: float, sampling_rate: float, amplitude: float = 1.0) -> tuple[np.ndarray, np.ndarray, int]:
    if bit_rate <= 0:
        raise ValueError("Bit rate must be greater than zero.")
    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than zero.")

    samples_per_bit = max(16, int(round(sampling_rate / bit_rate)))
    total_samples = len(bitstream) * samples_per_bit
    time_axis = np.arange(total_samples) / sampling_rate
    signal = np.zeros(total_samples)

    for idx, bit in enumerate(bitstream):
        start = idx * samples_per_bit
        end = start + samples_per_bit
        segment_time = time_axis[start:end]
        frequency = f1 if bit == "1" else f0
        signal[start:end] = amplitude * np.cos(2 * np.pi * frequency * segment_time)

    return signal, time_axis, samples_per_bit
