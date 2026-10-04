from __future__ import annotations

import numpy as np

from src.binary import bits_to_integer
from src.channel import awgn_channel
from src.fsk_demodulation import demodulate_bfsk
from src.fsk_modulation import generate_bfsk_signal
from src.metrics import compute_ber


def run_snr_sweep(bitstream: str, f0: float, f1: float, bit_rate: float, sampling_rate: float, snr_values: list[float]) -> list[dict]:
    results = []
    for snr_db in snr_values:
        modulated_signal, _, _ = generate_bfsk_signal(bitstream, f0, f1, bit_rate, sampling_rate)
        noisy_signal, _, _, _, _ = awgn_channel(modulated_signal, snr_db)
        recovered_bits, _ = demodulate_bfsk(noisy_signal, bitstream, f0, f1, bit_rate, sampling_rate)
        ber, bit_errors, total_bits = compute_ber(bitstream, recovered_bits)
        results.append({
            "snr_db": float(snr_db),
            "ber": float(ber),
            "bit_errors": int(bit_errors),
            "total_bits": int(total_bits),
        })
    return results


def decode_recovered_sensor_data(recovered_bits: str) -> tuple[int, int]:
    if len(recovered_bits) < 16:
        raise ValueError("Recovered bitstream is too short to decode the sensor payload.")
    temp_bits = recovered_bits[:8]
    humidity_bits = recovered_bits[8:16]
    return bits_to_integer(temp_bits), bits_to_integer(humidity_bits)


def expected_frequencies(f0: float, f1: float) -> tuple[float, float]:
    return float(f0), float(f1)
