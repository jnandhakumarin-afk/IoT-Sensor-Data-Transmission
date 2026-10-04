import unittest
import numpy as np

from src.sensor import validate_temperature, validate_humidity, generate_sensor_reading, formatted_timestamp
from src.binary import int_to_8bit, temperature_to_bits, humidity_to_bits, combine_sensor_bits, recover_sensor_values
from src.fsk_modulation import generate_bfsk_signal
from src.channel import awgn_channel, actual_snr_db
from src.fsk_demodulation import demodulate_bfsk
from src.metrics import compute_ber, compute_baud_rate, compute_bandwidth, transmission_quality
from src.analysis import run_snr_sweep, decode_recovered_sensor_data
from src.visualization import (
    plot_bfsk_waveform,
    plot_noisy_signal,
    plot_bit_comparison,
    plot_fft_spectrum,
    plot_ber_curve,
)


class TestFSKSystem(unittest.TestCase):

    def test_01_sensor_validation(self):
        self.assertTrue(validate_temperature(0.0))
        self.assertTrue(validate_temperature(100.0))
        self.assertTrue(validate_temperature(28.5))
        self.assertFalse(validate_temperature(-5.0))
        self.assertFalse(validate_temperature(105.0))

        self.assertTrue(validate_humidity(0.0))
        self.assertTrue(validate_humidity(100.0))
        self.assertTrue(validate_humidity(65.0))
        self.assertFalse(validate_humidity(-1.0))
        self.assertFalse(validate_humidity(101.0))

        temp, hum = generate_sensor_reading()
        self.assertTrue(20.0 <= temp <= 40.0)
        self.assertTrue(40.0 <= hum <= 90.0)
        self.assertTrue(len(formatted_timestamp()) > 10)

    def test_02_binary_framing(self):
        bits = combine_sensor_bits(28.0, 65.0)
        self.assertEqual(len(bits), 16)
        t_bits = bits[:8]
        h_bits = bits[8:]
        self.assertEqual(int(t_bits, 2), 28)
        self.assertEqual(int(h_bits, 2), 65)

        dec_t, dec_h = recover_sensor_values(bits)
        self.assertEqual(dec_t, 28)
        self.assertEqual(dec_h, 65)

    def test_03_bfsk_modulation(self):
        bitstream = "0001110001000001"
        f0, f1, bit_rate, fs = 1000.0, 2000.0, 10.0, 10000.0
        signal, time_axis, _ = generate_bfsk_signal(bitstream, f0, f1, bit_rate, fs)
        expected_samples = int(len(bitstream) * (fs / bit_rate))
        self.assertEqual(len(signal), expected_samples)
        self.assertEqual(len(time_axis), expected_samples)
        self.assertTrue(np.max(np.abs(signal)) <= 1.05)

    def test_04_awgn_channel(self):
        bitstream = "0001110001000001"
        signal, _, _ = generate_bfsk_signal(bitstream, 1000.0, 2000.0, 10.0, 10000.0)
        noisy, noise, sp, np_power, _ = awgn_channel(signal, snr_db=15.0)
        self.assertEqual(len(noisy), len(signal))
        self.assertTrue(sp > 0)
        self.assertTrue(np_power > 0)
        measured_snr = actual_snr_db(signal, noise)
        self.assertTrue(10.0 <= measured_snr <= 20.0)

    def test_05_demodulation_and_ber(self):
        bitstream = "0001110001000001"
        signal, _, _ = generate_bfsk_signal(bitstream, 1000.0, 2000.0, 10.0, 10000.0)
        noisy, _, _, _, _ = awgn_channel(signal, snr_db=25.0)
        recovered_bits, _ = demodulate_bfsk(noisy, bitstream, 1000.0, 2000.0, 10.0, 10000.0)
        self.assertEqual(len(recovered_bits), len(bitstream))
        ber, bit_errors, total_bits = compute_ber(bitstream, recovered_bits)
        self.assertEqual(total_bits, 16)
        self.assertEqual(bit_errors, 0)
        self.assertEqual(ber, 0.0)

    def test_06_metrics(self):
        self.assertEqual(compute_baud_rate(10.0), 10.0)
        self.assertEqual(compute_bandwidth(1000.0, 2000.0, 10.0), 1010.0)
        self.assertEqual(transmission_quality(0.0, 20.0), "Excellent")
        self.assertEqual(transmission_quality(0.005, 12.0), "Good")
        self.assertEqual(transmission_quality(0.04, 6.0), "Moderate")
        self.assertEqual(transmission_quality(0.15, 2.0), "Poor")

    def test_07_analysis_and_snr_sweep(self):
        bitstream = "0001110001000001"
        results = run_snr_sweep(bitstream, 1000.0, 2000.0, 10.0, 10000.0, [0, 10, 20])
        self.assertEqual(len(results), 3)
        self.assertIn("ber", results[0])
        self.assertIn("snr_db", results[0])

        t, h = decode_recovered_sensor_data(bitstream)
        self.assertEqual(t, 28)
        self.assertEqual(h, 65)

    def test_08_visualizations_render(self):
        bitstream = "0001110001000001"
        signal, time_axis, _ = generate_bfsk_signal(bitstream, 1000.0, 2000.0, 10.0, 10000.0)
        noisy, _, _, _, _ = awgn_channel(signal, 15.0)

        fig1 = plot_bfsk_waveform(time_axis[:500], signal[:500])
        self.assertIsNotNone(fig1)

        fig2 = plot_noisy_signal(time_axis[:500], signal[:500], noisy[:500])
        self.assertIsNotNone(fig2)

        fig3 = plot_bit_comparison(bitstream, bitstream)
        self.assertIsNotNone(fig3)

        fig4 = plot_fft_spectrum(signal, 10000.0, 1000.0, 2000.0)
        self.assertIsNotNone(fig4)

        results = [{"snr_db": 0, "ber": 0.2}, {"snr_db": 10, "ber": 0.0}]
        fig5 = plot_ber_curve(results)
        self.assertIsNotNone(fig5)


if __name__ == "__main__":
    unittest.main()
