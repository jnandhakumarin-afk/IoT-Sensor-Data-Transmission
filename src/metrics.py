from __future__ import annotations


def compute_ber(transmitted_bits: str, recovered_bits: str) -> tuple[float, int, int]:
    if len(transmitted_bits) != len(recovered_bits):
        total_bits = min(len(transmitted_bits), len(recovered_bits))
        transmitted_bits = transmitted_bits[:total_bits]
        recovered_bits = recovered_bits[:total_bits]

    bit_errors = sum(1 for tx, rx in zip(transmitted_bits, recovered_bits) if tx != rx)
    total_bits = len(transmitted_bits)
    ber = (bit_errors / total_bits) if total_bits else 0.0
    return ber, bit_errors, total_bits


def compute_baud_rate(bit_rate: float) -> float:
    return float(bit_rate)


def compute_bandwidth(f0: float, f1: float, bit_rate: float) -> float:
    return abs(f1 - f0) + bit_rate


def transmission_quality(ber: float, snr_db: float) -> str:
    if ber == 0 and snr_db >= 18:
        return "Excellent"
    if ber <= 0.01 and snr_db >= 10:
        return "Good"
    if ber <= 0.05 and snr_db >= 4:
        return "Moderate"
    return "Poor"
