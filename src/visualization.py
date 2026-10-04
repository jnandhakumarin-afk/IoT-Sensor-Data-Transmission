from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Strict Times New Roman font enforcement across all charts
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "serif"]


def plot_bfsk_waveform(time_axis: np.ndarray, signal: np.ndarray, title: str = "BFSK Modulated Signal"):
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(time_axis, signal, color="#00A8FF", linewidth=1.8)
    ax.set_title(title, color="#F8FAFC", fontsize=12)
    ax.set_xlabel("Time (seconds)", color="#94A3B8")
    ax.set_ylabel("Amplitude", color="#94A3B8")
    ax.tick_params(colors="#94A3B8")
    ax.set_facecolor("#071426")
    fig.patch.set_facecolor("#071426")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    for spine in ax.spines.values():
        spine.set_color("#173B5F")
    return fig


def plot_noisy_signal(time_axis: np.ndarray, original: np.ndarray, noisy: np.ndarray):
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(time_axis, original, color="#00A8FF", linewidth=1.1, alpha=0.9, label="Original BFSK")
    ax.plot(time_axis, noisy, color="#22C55E", linewidth=1.3, alpha=0.8, label="Noisy Received")
    ax.set_title("Wireless Channel: Transmitted vs Noisy Received Signal", color="#F8FAFC", fontsize=12)
    ax.set_xlabel("Time (seconds)", color="#94A3B8")
    ax.set_ylabel("Amplitude", color="#94A3B8")
    ax.legend(facecolor="#071426", edgecolor="#173B5F", labelcolor="#F8FAFC")
    ax.tick_params(colors="#94A3B8")
    ax.set_facecolor("#071426")
    fig.patch.set_facecolor("#071426")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    for spine in ax.spines.values():
        spine.set_color("#173B5F")
    return fig


def plot_bit_comparison(transmitted_bits: str, recovered_bits: str):
    fig, ax = plt.subplots(figsize=(10, 3.2))
    x = list(range(len(transmitted_bits)))
    ax.plot(x, [int(bit) for bit in transmitted_bits], marker="o", linestyle="-", linewidth=1.5, color="#00A8FF", label="Transmitted")
    ax.plot(x, [int(bit) for bit in recovered_bits], marker="s", linestyle="-", linewidth=1.5, color="#22C55E", label="Recovered")
    ax.set_title("Transmitted vs Recovered Bits", color="#F8FAFC", fontsize=12)
    ax.set_xlabel("Bit Index", color="#94A3B8")
    ax.set_ylabel("Bit Value", color="#94A3B8")
    ax.tick_params(colors="#94A3B8")
    ax.set_facecolor("#071426")
    fig.patch.set_facecolor("#071426")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    ax.legend(facecolor="#071426", edgecolor="#173B5F", labelcolor="#F8FAFC")
    for spine in ax.spines.values():
        spine.set_color("#173B5F")
    return fig


def plot_fft_spectrum(signal: np.ndarray, sampling_rate: float, f0: float, f1: float):
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), d=1 / sampling_rate)
    magnitude = np.abs(spectrum)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(freqs, magnitude, color="#00D9FF", linewidth=1.5)
    ax.axvline(f0, color="#F59E0B", linestyle="--", linewidth=1.4, label=f"f0 = {f0} Hz")
    ax.axvline(f1, color="#EF4444", linestyle="--", linewidth=1.4, label=f"f1 = {f1} Hz")
    ax.set_title("FFT Frequency Spectrum", color="#F8FAFC", fontsize=12)
    ax.set_xlabel("Frequency (Hz)", color="#94A3B8")
    ax.set_ylabel("Magnitude", color="#94A3B8")
    ax.legend(facecolor="#071426", edgecolor="#173B5F", labelcolor="#F8FAFC")
    ax.tick_params(colors="#94A3B8")
    ax.set_facecolor("#071426")
    fig.patch.set_facecolor("#071426")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    for spine in ax.spines.values():
        spine.set_color("#173B5F")
    return fig


def plot_ber_curve(results: list[dict]):
    fig, ax = plt.subplots(figsize=(9, 4))
    snr_values = [item["snr_db"] for item in results]
    ber_values = [item["ber"] for item in results]
    ax.plot(snr_values, ber_values, marker="o", color="#00A8FF", linewidth=2)
    ax.set_xscale("linear")
    ax.set_yscale("log")
    ax.set_title("SNR vs BER Analysis", color="#F8FAFC", fontsize=12)
    ax.set_xlabel("SNR (dB)", color="#94A3B8")
    ax.set_ylabel("BER", color="#94A3B8")
    ax.tick_params(colors="#94A3B8")
    ax.set_facecolor("#071426")
    fig.patch.set_facecolor("#071426")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    for spine in ax.spines.values():
        spine.set_color("#173B5F")
    return fig
