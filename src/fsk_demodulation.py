from __future__ import annotations

import numpy as np


def demodulate_bfsk(noisy_signal: np.ndarray, bitstream: str, f0: float, f1: float, bit_rate: float, sampling_rate: float) -> tuple[str, np.ndarray]:
    if bit_rate <= 0 or sampling_rate <= 0:
        raise ValueError("Bit rate and sampling rate must be positive.")

    samples_per_bit = max(16, int(round(sampling_rate / bit_rate)))
    t_segment = np.arange(samples_per_bit) / sampling_rate
    ref0 = np.cos(2 * np.pi * f0 * t_segment)
    ref1 = np.cos(2 * np.pi * f1 * t_segment)

    recovered_bits = []
    scores = []

    for idx in range(len(bitstream)):
        start = idx * samples_per_bit
        end = start + samples_per_bit
        segment = noisy_signal[start:end]
        score0 = float(np.dot(segment, ref0))
        score1 = float(np.dot(segment, ref1))
        scores.append((score0, score1))
        recovered_bits.append("1" if score1 > score0 else "0")

    return "".join(recovered_bits), np.array(scores, dtype=float)
