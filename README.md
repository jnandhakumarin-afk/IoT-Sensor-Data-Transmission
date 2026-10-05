# IoT Sensor Data Transmission Using FSK over a Wireless Communication Link

[![Live App](https://img.shields.io/badge/🚀%20Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://iot-sensor-data-transmission.streamlit.app/)

> **🌐 Live App:** [https://iot-sensor-data-transmission.streamlit.app/](https://iot-sensor-data-transmission.streamlit.app/)

---

## Project Title
IoT Sensor Data Transmission Using FSK over a Wireless Communication Link

## Objective
This project simulates the end-to-end transmission of IoT sensor values through a wireless link using Binary Frequency Shift Keying (BFSK). It demonstrates how sensor readings are converted into digital data, modulated, transmitted through a noisy wireless channel, demodulated, and recovered for analysis.

## Problem Statement
Modern IoT systems rely on digital communication links to send sensor information efficiently and reliably. In a noisy wireless channel, the transmitted data can be distorted by additive white Gaussian noise (AWGN). This application demonstrates the practical behavior of a BFSK-based digital communication system, from sensor acquisition to recovered values and BER analysis.

## Architecture
The architecture is modular and organized under the `src` folder:

- `sensor.py`: generates and validates sensor readings.
- `binary.py`: converts float sensor values into 8-bit binary streams.
- `fsk_modulation.py`: generates the BFSK waveform.
- `channel.py`: adds AWGN and computes relevant signal/noise power metrics.
- `fsk_demodulation.py`: demodulates received bits using frequency correlation.
- `metrics.py`: calculates BER, baud rate, bandwidth, and transmission quality.
- `analysis.py`: performs the SNR sweep and recovery logic.
- `visualization.py`: creates charts for modulation, FFT, bit comparison, and BER curves.

## Communication Flow
Sensor → Temperature/Humidity → Binary Conversion → BFSK Modulation → Wireless Channel → AWGN → BFSK Demodulation → Recovered Binary → Recovered Sensor Data → BER/SNR Analysis.

## Technology Stack
- Python 3.10+
- Streamlit
- NumPy
- Matplotlib

## Features
- Simulated or manual sensor input
- 8-bit binary representation of temperature and humidity
- BFSK modulation with configurable carrier frequencies and bit rate
- AWGN noisy channel modeling
- Correlation-based BFSK demodulation
- Recovered sensor value reconstruction
- BER and SNR analysis
- FFT spectrum visualization
- SNR vs BER performance curve
- Responsive dark navy user interface

## BFSK Explanation
Binary Frequency Shift Keying uses two carrier frequencies to represent digital bits. In this project:

- Bit 0 → f0
- Bit 1 → f1

The transmitted signal is a sum of cosine waves with the selected carrier frequency for each bit interval.

## AWGN Explanation
AWGN is Additive White Gaussian Noise added to the signal in the channel. It models real-world wireless interference, making the link less reliable at lower SNR values. The noise is generated from a Gaussian distribution and scaled using the configured SNR.

## Demodulation Explanation
Each received bit segment is compared against the two reference frequencies used in BFSK. The signal is correlated with the f0 and f1 reference waveforms, and the stronger response determines whether the bit is 0 or 1.

## BER
Bit Error Rate (BER) is computed as:

BER = Number of bit errors / Total bits compared

This value measures how accurately the received bits match the transmitted bits.

## SNR
Signal-to-noise ratio is measured in dB using the relation:

SNR(dB) = 10 log10(Psignal / Pnoise)

## Bit Rate
Bit rate is the number of bits transmitted per second. In this system it is configured by the user and used in the signal generation and bandwidth approximation.

## Baud Rate
For binary FSK, the baud rate is equivalent to the bit rate, because each symbol carries one bit.

## Bandwidth
Approximate BFSK bandwidth is estimated as:

BW ≈ |f1 - f0| + Bit Rate

## FFT Analysis
The FFT spectrum highlights the main frequency components of the BFSK signal. The expected f0 and f1 tones should be visible in the frequency domain.

## SNR vs BER
The application performs a real SNR sweep across values from 0 dB to 30 dB. For each SNR level it transmits the signal over AWGN, demodulates it, and computes BER. The resulting curve is plotted on a log scale for BER.

## Installation
1. Open the project folder.
2. Create and activate a Python virtual environment if desired.
3. Install the required packages:

```bash
pip install -r requirements.txt
```

## Run Command
```bash
streamlit run app.py
```

## Folder Structure
```text
FSK/
├── app.py
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── sensor.py
│   ├── binary.py
│   ├── fsk_modulation.py
│   ├── channel.py
│   ├── fsk_demodulation.py
│   ├── metrics.py
│   ├── analysis.py
│   └── visualization.py
├── results/
│   └── .gitkeep
└── .streamlit/
```

## Screens and Pages
- Home
- Sensor Data
- Binary Conversion
- BFSK Modulation
- Wireless Channel
- Demodulation
- Signal Analysis
- Performance
- Final Result

## Expected Output
The app provides a complete simulation flow where sensor data is transformed into binary symbols, modulated with BFSK, transmitted through AWGN, demodulated, and finally recovered with BER and SNR metrics.

## Academic Relevance
This project is relevant to Analog and Digital Communication courses and demonstrates digital modulation, channel noise, signal recovery, and performance evaluation in a realistic laboratory-style interface.

## Future Enhancements
- Add M-ary FSK comparison
- Add channel coding and error correction
- Add phase shift keying comparison
- Add export of performance reports
- Add more advanced signal plotting and waveform controls
