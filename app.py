from __future__ import annotations

import time
from datetime import datetime

import numpy as np
import streamlit as st

from src.analysis import decode_recovered_sensor_data, run_snr_sweep
from src.binary import combine_sensor_bits, recover_sensor_values
from src.channel import actual_snr_db, awgn_channel
from src.fsk_demodulation import demodulate_bfsk
from src.fsk_modulation import generate_bfsk_signal
from src.metrics import compute_bandwidth, compute_ber, compute_baud_rate, transmission_quality
from src.sensor import formatted_timestamp, generate_sensor_reading, validate_humidity, validate_temperature
from src.visualization import (
    plot_ber_curve,
    plot_bit_comparison,
    plot_bfsk_waveform,
    plot_fft_spectrum,
    plot_noisy_signal,
)

PROJECT_TITLE = "IoT Sensor Data Transmission Using FSK"
PROJECT_SUBTITLE = "Wireless Digital Communication Simulator"

st.set_page_config(
    page_title=PROJECT_TITLE,
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = [
    "HOME",
    "SENSOR DATA",
    "BINARY CONVERSION",
    "BFSK MODULATION",
    "WIRELESS CHANNEL",
    "DEMODULATION",
    "SIGNAL ANALYSIS",
    "PERFORMANCE",
    "FINAL RESULT",
]

DEFAULTS = {
    "temperature": 28.5,
    "humidity": 65.0,
    "reading_mode": "MANUAL",
    "binary_data": "0001110001000001",
    "f0": 1000.0,
    "f1": 2000.0,
    "bit_rate": 10.0,
    "sampling_rate": 10000.0,
    "snr": 14.0,
    "modulated_signal": np.array([], dtype=float),
    "noisy_signal": np.array([], dtype=float),
    "recovered_bits": "0001110001000001",
    "BER": 0.0,
    "SNR_results": [],
    "analysis_results": {},
    "simulation_ready": False,
    "timestamp": formatted_timestamp(),
    "signal_power": 0.0,
    "noise_power": 0.0,
    "configured_snr": 14.0,
    "actual_snr": 14.0,
    "bit_errors": 0,
    "total_bits": 16,
    "bandwidth": 1010.0,
    "baud_rate": 10.0,
    "quality": "Excellent",
    "recovered_temperature": 28,
    "recovered_humidity": 65,
}


def init_session_state():
    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value)

    if "page" not in st.session_state:
        st.session_state.page = "HOME"

    if "splash_time" not in st.session_state:
        st.session_state.splash_time = time.time() + 1.2


def render_html(html_str: str) -> None:
    """Safely render HTML without CommonMark treating indented blocks or blank lines as code blocks."""
    cleaned = " ".join(line.strip() for line in html_str.strip().splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


def inject_custom_css():
    st.markdown(
        """
        <style>
        /* ========================================================
           100% STRICT TIMES NEW ROMAN - EVERY SINGLE WORD & ELEMENT
           ======================================================== */
        @import url('https://fonts.googleapis.com/css2?family=Tinos:ital,wght@0,400;0,700;1,400;1,700&display=swap');

        *, *::before, *::after, html, body, div, span, applet, object, iframe,
        h1, h2, h3, h4, h5, h6, p, blockquote, pre, a, abbr, acronym, address,
        big, cite, code, del, dfn, em, img, ins, kbd, q, s, samp, small, strike,
        strong, sub, sup, tt, var, b, u, i, center, dl, dt, dd, ol, ul, li,
        fieldset, form, label, legend, table, caption, tbody, tfoot, thead, tr, th, td,
        article, aside, canvas, details, embed, figure, figcaption, footer, header,
        hgroup, menu, nav, output, ruby, section, summary, time, mark, audio, video,
        input, button, select, textarea, [class*="css"], [data-testid="stAppViewContainer"],
        [data-testid="stApp"], [data-testid="stSidebar"], [data-testid="stSidebar"] *,
        [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] *,
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"],
        .stButton, .stButton button, .stNumberInput input, .stTextInput input,
        .stRadio, .stRadio label, div[role="radiogroup"] label {
            font-family: 'Times New Roman', 'Tinos', Times, Georgia, serif !important;
            -webkit-font-smoothing: antialiased;
        }

        /* Preserve Google Material Icons / Symbols if loaded */
        .material-icons,
        .material-symbols-rounded,
        .material-symbols-outlined,
        [data-testid="stIconMaterial"] {
            font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
        }

        /* Dark High-Tech 3D Radial Background */
        html, body, [data-testid="stAppViewContainer"], .stApp {
            background-color: #030712 !important;
            background-image: 
                radial-gradient(circle at 50% 0%, rgba(0, 168, 255, 0.12) 0%, transparent 50%),
                radial-gradient(circle at 100% 50%, rgba(0, 217, 255, 0.05) 0%, transparent 40%),
                linear-gradient(180deg, #040914 0%, #030712 100%) !important;
            color: #F8FAFC !important;
        }

        /* Scalable Container Layout */
        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 2.5rem !important;
            padding-left: clamp(1rem, 3.5vw, 3rem) !important;
            padding-right: clamp(1rem, 3.5vw, 3rem) !important;
            max-width: 1420px !important;
        }

        /* 3D Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #071324 0%, #040A14 100%) !important;
            border-right: 1px solid rgba(0, 168, 255, 0.22) !important;
            box-shadow: 4px 0 24px rgba(0, 0, 0, 0.6) !important;
        }

        /* ========================================================
           SIDEBAR TOGGLE ARROW FIX - REPLACES key_double... WITH « / »
           ======================================================== */
        [data-testid="stSidebarCollapseButton"],
        [data-testid="collapsedControl"],
        [data-testid="stSidebarCollapseButton"] button,
        [data-testid="collapsedControl"] button,
        button[data-testid="baseButton-headerNoPadding"],
        button[aria-label*="sidebar" i],
        button[aria-label*="Sidebar" i] {
            position: relative !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            background: rgba(0, 168, 255, 0.12) !important;
            border: 1px solid rgba(0, 168, 255, 0.35) !important;
            border-radius: 8px !important;
            min-width: 2.2rem !important;
            height: 2.2rem !important;
            cursor: pointer !important;
            transition: all 0.2s ease !important;
        }

        [data-testid="stSidebarCollapseButton"] button:hover,
        [data-testid="collapsedControl"] button:hover,
        button[aria-label*="sidebar" i]:hover {
            background: rgba(0, 168, 255, 0.25) !important;
            border-color: #00D9FF !important;
            box-shadow: 0 0 12px rgba(0, 217, 255, 0.35) !important;
        }

        /* Completely hide the raw text ligature like keyboard_double_arrow_left */
        [data-testid="stSidebarCollapseButton"] button *,
        [data-testid="collapsedControl"] button *,
        button[aria-label*="sidebar" i] * {
            font-size: 0px !important;
            color: transparent !important;
            visibility: hidden !important;
            width: 0 !important;
            height: 0 !important;
            line-height: 0 !important;
            overflow: hidden !important;
            display: none !important;
        }

        /* Inject clean, glowing cyan double arrow « (collapse sidebar) */
        [data-testid="stSidebarCollapseButton"] button::after,
        button[aria-label*="close sidebar" i]::after,
        button[aria-label*="collapse sidebar" i]::after {
            content: "«" !important;
            font-size: 1.45rem !important;
            font-weight: 700 !important;
            color: #00D9FF !important;
            visibility: visible !important;
            line-height: 1 !important;
            display: block !important;
            font-family: 'Times New Roman', Times, serif !important;
        }

        /* Inject clean, glowing cyan double arrow » (expand sidebar) */
        [data-testid="collapsedControl"]::after,
        [data-testid="collapsedControl"] button::after,
        button[aria-label*="open sidebar" i]::after,
        button[aria-label*="expand sidebar" i]::after {
            content: "»" !important;
            font-size: 1.45rem !important;
            font-weight: 700 !important;
            color: #00D9FF !important;
            visibility: visible !important;
            line-height: 1 !important;
            display: block !important;
            font-family: 'Times New Roman', Times, serif !important;
        }

        /* Sidebar Radio Navigation Items */
        .stRadio > div {
            gap: 0.35rem !important;
        }

        div[role="radiogroup"] > label {
            background: rgba(11, 28, 51, 0.55) !important;
            border: 1px solid rgba(23, 59, 95, 0.8) !important;
            border-radius: 9px !important;
            padding: 0.45rem 0.75rem !important;
            transition: all 0.2s ease-in-out !important;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.3) !important;
        }

        div[role="radiogroup"] > label:hover {
            background: rgba(0, 168, 255, 0.14) !important;
            border-color: #00A8FF !important;
            transform: translateX(3px) !important;
        }

        div[role="radiogroup"] > label[data-checked="true"],
        div[role="radiogroup"] > label:has(input:checked) {
            background: linear-gradient(90deg, rgba(0, 168, 255, 0.25) 0%, rgba(11, 28, 51, 0.95) 100%) !important;
            border: 1px solid #00D9FF !important;
            box-shadow: 0 0 12px rgba(0, 217, 255, 0.25) !important;
        }

        /* 3D Tactile Buttons */
        .stButton > button {
            background: linear-gradient(180deg, #00A8FF 0%, #006BB3 100%) !important;
            color: #FFFFFF !important;
            font-size: 1.0rem !important;
            font-weight: 700 !important;
            padding: 0.65rem 1.4rem !important;
            border-radius: 10px !important;
            border: 1px solid rgba(255, 255, 255, 0.22) !important;
            border-bottom: 3.5px solid #004D80 !important;
            box-shadow: 0 6px 16px rgba(0, 168, 255, 0.3), 0 3px 6px rgba(0, 0, 0, 0.5) !important;
            transition: all 0.16s ease-in-out !important;
            cursor: pointer !important;
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.4) !important;
            letter-spacing: 0.03em !important;
        }

        .stButton > button:hover {
            background: linear-gradient(180deg, #24C6FF 0%, #0088DD 100%) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 10px 24px rgba(0, 217, 255, 0.4), 0 4px 10px rgba(0, 0, 0, 0.6) !important;
            border-bottom: 3.5px solid #005B99 !important;
        }

        .stButton > button:active {
            transform: translateY(2px) !important;
            border-bottom: 1px solid #004D80 !important;
            box-shadow: 0 2px 6px rgba(0, 168, 255, 0.25) !important;
        }

        /* 3D Glassmorphic Cards */
        .glass-card-3d {
            background: linear-gradient(135deg, rgba(14, 34, 61, 0.85) 0%, rgba(6, 17, 33, 0.95) 100%);
            border: 1px solid rgba(0, 168, 255, 0.22);
            border-top: 1px solid rgba(255, 255, 255, 0.18);
            border-radius: 14px;
            padding: 1.25rem;
            box-shadow: 
                0 12px 28px rgba(0, 0, 0, 0.6),
                0 0 14px rgba(0, 168, 255, 0.08),
                inset 0 1px 1px rgba(255, 255, 255, 0.12);
            position: relative;
            overflow: hidden;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .glass-card-3d:hover {
            border-color: rgba(0, 217, 255, 0.4);
            box-shadow: 
                0 16px 36px rgba(0, 0, 0, 0.7),
                0 0 20px rgba(0, 217, 255, 0.18),
                inset 0 1px 1px rgba(255, 255, 255, 0.2);
        }

        /* Elegant Data Stream Display (Non-coding, Professional Times New Roman Pill Box) */
        .telemetry-stream-box {
            background: linear-gradient(180deg, #061324 0%, #030A14 100%);
            border: 1px solid #1D436D;
            border-radius: 9px;
            padding: 0.65rem 1.0rem;
            color: #00D9FF;
            font-size: 1.15rem;
            font-weight: 700;
            letter-spacing: 0.18em;
            text-align: center;
            box-shadow: inset 0 2px 6px rgba(0, 0, 0, 0.7);
        }

        /* Input Controls */
        .stNumberInput input, .stTextInput input {
            background: #081628 !important;
            border: 1px solid #1A3E66 !important;
            border-radius: 9px !important;
            color: #F8FAFC !important;
            font-size: 1.0rem !important;
            padding: 0.55rem 0.85rem !important;
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.5) !important;
        }

        .stNumberInput input:focus, .stTextInput input:focus {
            border-color: #00D9FF !important;
            box-shadow: 0 0 8px rgba(0, 217, 255, 0.3) !important;
        }

        .stNumberInput label, .stTextInput label {
            font-size: 0.92rem !important;
            font-weight: 700 !important;
            color: #CBD5E1 !important;
            margin-bottom: 0.3rem !important;
        }

        /* Alerts */
        .stSuccess {
            background: rgba(16, 185, 129, 0.14) !important;
            border: 1px solid #10B981 !important;
            color: #A7F3D0 !important;
            border-radius: 10px !important;
        }

        .stWarning {
            background: rgba(245, 158, 11, 0.14) !important;
            border: 1px solid #F59E0B !important;
            color: #FDE68A !important;
            border-radius: 10px !important;
        }

        .stError {
            background: rgba(239, 68, 68, 0.14) !important;
            border: 1px solid #EF4444 !important;
            color: #FECACA !important;
            border-radius: 10px !important;
        }

        /* Responsive Breakpoints */
        @media (max-width: 900px) {
            .block-container {
                padding-left: 0.85rem !important;
                padding-right: 0.85rem !important;
            }
            .stButton > button {
                width: 100% !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_splash_screen():
    render_html(
        f"""
        <div style="min-height: 82vh; display:flex; align-items:center; justify-content:center; text-align:center;">
            <div style="
                border: 1px solid rgba(0, 168, 255, 0.35);
                border-top: 1px solid rgba(255, 255, 255, 0.25);
                background: linear-gradient(145deg, #0A1E38, #050D1A);
                border-radius: 22px;
                padding: clamp(1.8rem, 4.5vw, 3.0rem);
                box-shadow: 0 18px 45px rgba(0, 0, 0, 0.8), 0 0 30px rgba(0, 168, 255, 0.22);
                max-width: 720px;
                width: 92%;
            ">
                <div style="font-size: 2.5rem; margin-bottom: 0.6rem;">📡</div>
                <div style="font-size: clamp(1.7rem, 3.8vw, 2.3rem); font-weight: 700; color: #FFFFFF; letter-spacing: 0.03em; text-shadow: 0 2px 10px rgba(0, 168, 255, 0.5);">
                    {PROJECT_TITLE}
                </div>
                <div style="font-size: 1.05rem; color: #00D9FF; letter-spacing: 0.12em; margin-top: 0.5rem; font-weight: 700;">
                    {PROJECT_SUBTITLE.upper()}
                </div>
                <div style="height: 1px; background: linear-gradient(90deg, transparent, #00A8FF, transparent); margin: 1.3rem 0;"></div>
                <div style="font-size: 1.0rem; color: #CBD5E1; line-height: 1.6;">
                    Sensor Acquisition ➔ 8-Bit Digital Framing ➔ BFSK Modulation ➔ AWGN Channel ➔ Coherent Demodulation ➔ Data Recovery
                </div>
            </div>
        </div>
        """
    )
    time.sleep(0.3)


def validate_config():
    if st.session_state.f0 == st.session_state.f1:
        st.error("Carrier frequency f0 must differ from f1.")
        return False
    if st.session_state.sampling_rate <= max(st.session_state.f0, st.session_state.f1):
        st.error("Sampling rate must be greater than highest carrier frequency.")
        return False
    if st.session_state.bit_rate <= 0:
        st.error("Bit rate must be greater than zero.")
        return False
    if not validate_temperature(st.session_state.temperature):
        st.error("Temperature must be between 0 and 100 °C.")
        return False
    if not validate_humidity(st.session_state.humidity):
        st.error("Humidity must be between 0 and 100 %.")
        return False
    if st.session_state.snr < -20 or st.session_state.snr > 50:
        st.error("SNR should be between -20 dB and 50 dB.")
        return False
    return True


def run_full_simulation():
    if not validate_config():
        return False

    temperature = float(st.session_state.temperature)
    humidity = float(st.session_state.humidity)

    bitstream = combine_sensor_bits(temperature, humidity)
    st.session_state.binary_data = bitstream

    modulated_signal, time_axis, samples_per_bit = generate_bfsk_signal(
        bitstream=bitstream,
        f0=float(st.session_state.f0),
        f1=float(st.session_state.f1),
        bit_rate=float(st.session_state.bit_rate),
        sampling_rate=float(st.session_state.sampling_rate),
    )
    st.session_state.modulated_signal = modulated_signal

    noisy_signal, noise, signal_power, noise_power, noise_sigma = awgn_channel(
        modulated_signal,
        float(st.session_state.snr),
    )
    st.session_state.noisy_signal = noisy_signal
    st.session_state.signal_power = signal_power
    st.session_state.noise_power = noise_power
    st.session_state.configured_snr = float(st.session_state.snr)
    st.session_state.actual_snr = actual_snr_db(modulated_signal, noise)

    recovered_bits, scores = demodulate_bfsk(
        noisy_signal,
        bitstream,
        float(st.session_state.f0),
        float(st.session_state.f1),
        float(st.session_state.bit_rate),
        float(st.session_state.sampling_rate),
    )
    st.session_state.recovered_bits = recovered_bits

    ber, bit_errors, total_bits = compute_ber(bitstream, recovered_bits)
    st.session_state.BER = ber
    st.session_state.bit_errors = bit_errors
    st.session_state.total_bits = total_bits

    st.session_state.bandwidth = compute_bandwidth(
        float(st.session_state.f0),
        float(st.session_state.f1),
        float(st.session_state.bit_rate),
    )
    st.session_state.baud_rate = compute_baud_rate(float(st.session_state.bit_rate))
    st.session_state.quality = transmission_quality(ber, st.session_state.actual_snr)

    rec_temp, rec_hum = decode_recovered_sensor_data(recovered_bits)
    st.session_state.recovered_temperature = rec_temp
    st.session_state.recovered_humidity = rec_hum

    snr_sweep_results = run_snr_sweep(
        bitstream=bitstream,
        f0=float(st.session_state.f0),
        f1=float(st.session_state.f1),
        bit_rate=float(st.session_state.bit_rate),
        sampling_rate=float(st.session_state.sampling_rate),
        snr_values=list(range(0, 31, 2)),
    )
    st.session_state.SNR_results = snr_sweep_results

    st.session_state.analysis_results = {
        "time_axis": time_axis,
        "signal": modulated_signal,
        "bitstream": bitstream,
    }

    st.session_state.simulation_ready = True
    return True


def render_sidebar():
    with st.sidebar:
        render_html(
            f"""
            <div style="padding: 0.4rem 0.2rem;">
                <div style="font-size: 1.25rem; font-weight: 700; color: #FFFFFF; line-height: 1.3;">
                    📡 {PROJECT_TITLE}
                </div>
                <div style="font-size: 0.78rem; color: #00D9FF; letter-spacing: 0.12em; margin-top: 0.25rem; font-weight: 700;">
                    {PROJECT_SUBTITLE.upper()}
                </div>
            </div>
            """
        )

        render_html("<hr style='border: none; border-top: 1px solid rgba(23, 59, 95, 0.6); margin: 0.8rem 0;'>")
        render_html("<div style='font-size: 0.82rem; font-weight: 700; color: #94A3B8; letter-spacing: 0.08em; margin-bottom: 0.4rem;'>SIMULATION STAGES</div>")

        page_selected = st.radio(
            "Navigation",
            PAGES,
            index=PAGES.index(st.session_state.page),
            label_visibility="collapsed",
        )
        if page_selected != st.session_state.page:
            st.session_state.page = page_selected
            st.rerun()

        render_html("<hr style='border: none; border-top: 1px solid rgba(23, 59, 95, 0.6); margin: 0.8rem 0;'>")
        render_html("<div style='font-size: 0.82rem; font-weight: 700; color: #94A3B8; letter-spacing: 0.08em; margin-bottom: 0.4rem;'>CHANNEL & RF CONFIG</div>")

        st.number_input("Carrier f0 (Hz) [Bit 0]", min_value=10.0, step=100.0, key="f0")
        st.number_input("Carrier f1 (Hz) [Bit 1]", min_value=20.0, step=100.0, key="f1")
        st.number_input("Bit Rate (bps)", min_value=1.0, step=5.0, key="bit_rate")
        st.number_input("Sampling Rate (Hz)", min_value=100.0, step=500.0, key="sampling_rate")
        st.number_input("Wireless SNR (dB)", min_value=-20.0, max_value=50.0, step=1.0, key="snr")

        render_html("<div style='height: 8px;'></div>")
        if st.button("⚡ RE-RUN SIMULATION", use_container_width=True, key="sidebar_rerun"):
            run_full_simulation()
            st.rerun()


def render_page_nav(prev_label: str, next_label: str, prev_page: str, next_page: str):
    render_html("<div style='height: 22px;'></div>")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button(f"← PREVIOUS: {prev_label}", use_container_width=True, key=f"nav_prev_{prev_page}"):
            st.session_state.page = prev_page
            st.rerun()
    with col2:
        if st.button(f"NEXT: {next_label} →", use_container_width=True, key=f"nav_next_{next_page}"):
            st.session_state.page = next_page
            st.rerun()


def render_sensor_status_grid():
    """Renders CURRENT SENSOR STATUS with proper, professional, well-proportioned values (not big)."""
    cols = st.columns(4)
    with cols[0]:
        render_html(
            f"""
            <div class="glass-card-3d" style="padding: 0.95rem 1.15rem;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 0.82rem; font-weight: 700; color: #94A3B8; letter-spacing: 0.06em;">🌡️ TEMPERATURE</span>
                    <span style="background: rgba(0, 168, 255, 0.15); color: #00D9FF; font-size: 0.72rem; padding: 0.15rem 0.45rem; border-radius: 5px; font-weight: 700; border: 1px solid rgba(0, 168, 255, 0.4);">ACTIVE</span>
                </div>
                <div style="font-size: 1.25rem; font-weight: 700; color: #FFFFFF; margin: 0.35rem 0 0.15rem 0;">
                    {st.session_state.temperature:.2f} <span style="font-size: 0.95rem; color: #94A3B8; font-weight: 400;">°C</span>
                </div>
                <div style="font-size: 0.75rem; color: #10B981; font-weight: 700;">● Nominal Calibration</div>
            </div>
            """
        )
    with cols[1]:
        render_html(
            f"""
            <div class="glass-card-3d" style="padding: 0.95rem 1.15rem;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 0.82rem; font-weight: 700; color: #94A3B8; letter-spacing: 0.06em;">💧 HUMIDITY</span>
                    <span style="background: rgba(16, 185, 129, 0.15); color: #10B981; font-size: 0.72rem; padding: 0.15rem 0.45rem; border-radius: 5px; font-weight: 700; border: 1px solid rgba(16, 185, 129, 0.4);">ACTIVE</span>
                </div>
                <div style="font-size: 1.25rem; font-weight: 700; color: #FFFFFF; margin: 0.35rem 0 0.15rem 0;">
                    {st.session_state.humidity:.2f} <span style="font-size: 0.95rem; color: #94A3B8; font-weight: 400;">%</span>
                </div>
                <div style="font-size: 0.75rem; color: #10B981; font-weight: 700;">● Ambient Environment</div>
            </div>
            """
        )
    with cols[2]:
        render_html(
            f"""
            <div class="glass-card-3d" style="padding: 0.95rem 1.15rem;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 0.82rem; font-weight: 700; color: #94A3B8; letter-spacing: 0.06em;">🎛️ ACQUISITION MODE</span>
                    <span style="background: rgba(245, 158, 11, 0.15); color: #F59E0B; font-size: 0.72rem; padding: 0.15rem 0.45rem; border-radius: 5px; font-weight: 700; border: 1px solid rgba(245, 158, 11, 0.4);">INPUT</span>
                </div>
                <div style="font-size: 1.22rem; font-weight: 700; color: #00D9FF; margin: 0.35rem 0 0.15rem 0;">
                    {st.session_state.reading_mode}
                </div>
                <div style="font-size: 0.75rem; color: #94A3B8;">● Dual-channel source</div>
            </div>
            """
        )
    with cols[3]:
        render_html(
            f"""
            <div class="glass-card-3d" style="padding: 0.95rem 1.15rem;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 0.82rem; font-weight: 700; color: #94A3B8; letter-spacing: 0.06em;">🕒 TIMESTAMP</span>
                    <span style="background: rgba(0, 168, 255, 0.15); color: #00A8FF; font-size: 0.72rem; padding: 0.15rem 0.45rem; border-radius: 5px; font-weight: 700; border: 1px solid rgba(0, 168, 255, 0.4);">SYNCED</span>
                </div>
                <div style="font-size: 1.05rem; font-weight: 700; color: #FFFFFF; margin: 0.45rem 0 0.2rem 0; line-height: 1.25;">
                    {st.session_state.timestamp}
                </div>
                <div style="font-size: 0.75rem; color: #10B981; font-weight: 700;">● RTC Calibrated</div>
            </div>
            """
        )


def render_home():
    if not st.session_state.simulation_ready:
        run_full_simulation()

    # 3D Hero Section
    render_html(
        f"""
        <div class="glass-card-3d" style="padding: 2.0rem; text-align: center; margin-bottom: 1.6rem; background: linear-gradient(145deg, rgba(14, 38, 70, 0.9), rgba(5, 14, 27, 0.95));">
            <div style="display: inline-block; background: rgba(0, 168, 255, 0.15); border: 1px solid rgba(0, 168, 255, 0.5); padding: 0.3rem 1.0rem; border-radius: 20px; font-size: 0.82rem; font-weight: 700; color: #00D9FF; letter-spacing: 0.12em; margin-bottom: 0.8rem;">
                ✦ END-TO-END WIRELESS TELEMETRY SYSTEM ✦
            </div>
            <div style="font-size: clamp(1.9rem, 3.5vw, 2.7rem); font-weight: 700; color: #FFFFFF; line-height: 1.25; margin-bottom: 0.5rem; text-shadow: 0 4px 18px rgba(0, 168, 255, 0.45);">
                {PROJECT_TITLE}
            </div>
            <div style="font-size: 1.15rem; color: #CBD5E1; max-width: 820px; margin: 0.4rem auto 1.2rem auto; line-height: 1.6;">
                A high-fidelity digital communication system modeling the complete pipeline: sensor data acquisition, 
                8-bit digital encoding, BFSK signal modulation, noisy AWGN wireless transmission, coherent demodulation, and bit error rate analysis.
            </div>
        </div>
        """
    )

    # 3D Quick Action Buttons
    col_act1, col_act2 = st.columns([1, 1])
    with col_act1:
        if st.button("🚀 START SIMULATION WORKFLOW →", use_container_width=True, key="home_btn_start"):
            st.session_state.page = "SENSOR DATA"
            st.rerun()
    with col_act2:
        if st.button("📊 VIEW PERFORMANCE & BER CURVES →", use_container_width=True, key="home_btn_perf"):
            st.session_state.page = "PERFORMANCE"
            st.rerun()

    render_html("<div style='height: 16px;'></div>")

    # 3D Architecture Visual Diagram - Header
    render_html(
        """
        <div class="glass-card-3d" style="padding: 1.2rem 1.6rem; margin-bottom: 0.8rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem;">
                <div style="font-size: 1.2rem; font-weight: 700; color: #FFFFFF; letter-spacing: 0.04em;">
                    🌐 END-TO-END 3D SYSTEM ARCHITECTURE
                </div>
                <div style="font-size: 0.82rem; color: #00D9FF; font-weight: 700;">
                    BFSK TRANSMISSION PIPELINE
                </div>
            </div>
        </div>
        """
    )

    # 6-Stage Architecture Columns (Safe, Clean, Zero Code Block Glitch)
    arch_cols = st.columns(6)
    stages = [
        ("🌡️", "#00D9FF", "STAGE 01", "IoT Sensors", "Temp & Humidity Data", "rgba(0,168,255,0.3)"),
        ("🔢", "#00D9FF", "STAGE 02", "Binary Framing", "16-bit Payload Stream", "rgba(0,168,255,0.3)"),
        ("⚡", "#00D9FF", "STAGE 03", "BFSK Modulator", "Dual-tone f0 / f1 Carrier", "rgba(0,168,255,0.3)"),
        ("〰️", "#00D9FF", "STAGE 04", "AWGN Channel", "Gaussian Noise Model", "rgba(0,168,255,0.3)"),
        ("🎯", "#00D9FF", "STAGE 05", "BFSK Demodulator", "Coherent Correlator", "rgba(0,168,255,0.3)"),
        ("📊", "#10B981", "STAGE 06", "Data Recovery", "Restored Sensor Readings", "rgba(16,185,129,0.4)"),
    ]
    for col, (icon, label_color, stage, title, sub, border) in zip(arch_cols, stages):
        with col:
            render_html(
                f"""
                <div style="background: rgba(8, 22, 42, 0.8); border: 1px solid {border}; border-radius: 11px; padding: 1.0rem 0.75rem; text-align: center; height: 100%;">
                    <div style="font-size: 1.6rem; margin-bottom: 0.3rem;">{icon}</div>
                    <div style="font-size: 0.72rem; color: {label_color}; font-weight: 700; letter-spacing: 0.08em;">{stage}</div>
                    <div style="font-size: 1.0rem; font-weight: 700; color: #FFFFFF; margin-top: 0.2rem;">{title}</div>
                    <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 0.25rem;">{sub}</div>
                </div>
                """
            )

    render_html("<div style='height: 16px;'></div>")

    # Live System Telemetry Metrics
    render_html("<div style='font-size: 1.15rem; font-weight: 700; color: #FFFFFF; margin-bottom: 0.4rem;'>📡 LIVE TELEMETRY & SYSTEM METRICS</div>")
    metric_cols = st.columns(4)
    with metric_cols[0]:
        render_html(
            f"""
            <div class="glass-card-3d" style="padding: 1.0rem;">
                <div style="font-size: 0.78rem; color: #94A3B8; font-weight: 700;">TRANSMITTED TEMPERATURE</div>
                <div style="font-size: 1.35rem; font-weight: 700; color: #00D9FF; margin-top: 0.25rem;">{st.session_state.temperature:.1f} °C</div>
                <div style="font-size: 0.75rem; color: #10B981;">Recovered: {st.session_state.recovered_temperature} °C</div>
            </div>
            """
        )
    with metric_cols[1]:
        render_html(
            f"""
            <div class="glass-card-3d" style="padding: 1.0rem;">
                <div style="font-size: 0.78rem; color: #94A3B8; font-weight: 700;">TRANSMITTED HUMIDITY</div>
                <div style="font-size: 1.35rem; font-weight: 700; color: #10B981; margin-top: 0.25rem;">{st.session_state.humidity:.1f} %</div>
                <div style="font-size: 0.75rem; color: #10B981;">Recovered: {st.session_state.recovered_humidity} %</div>
            </div>
            """
        )
    with metric_cols[2]:
        render_html(
            f"""
            <div class="glass-card-3d" style="padding: 1.0rem;">
                <div style="font-size: 0.78rem; color: #94A3B8; font-weight: 700;">CHANNEL SNR</div>
                <div style="font-size: 1.35rem; font-weight: 700; color: #F59E0B; margin-top: 0.25rem;">{st.session_state.snr:.1f} dB</div>
                <div style="font-size: 0.75rem; color: #94A3B8;">Measured: {st.session_state.actual_snr:.2f} dB</div>
            </div>
            """
        )
    with metric_cols[3]:
        ber_color = "#10B981" if st.session_state.BER == 0 else "#EF4444"
        render_html(
            f"""
            <div class="glass-card-3d" style="padding: 1.0rem;">
                <div style="font-size: 0.78rem; color: #94A3B8; font-weight: 700;">BIT ERROR RATE (BER)</div>
                <div style="font-size: 1.35rem; font-weight: 700; color: {ber_color}; margin-top: 0.25rem;">{st.session_state.BER:.4f}</div>
                <div style="font-size: 0.75rem; color: {ber_color};">Errors: {st.session_state.bit_errors} / {st.session_state.total_bits} bits</div>
            </div>
            """
        )

    render_html("<div style='height: 18px;'></div>")

    # Technical Specifications Overview Card
    render_html(
        f"""
        <div class="glass-card-3d" style="padding: 1.4rem;">
            <div style="font-size: 1.15rem; font-weight: 700; color: #FFFFFF; margin-bottom: 0.75rem;">
                ⚙️ ACTIVE SYSTEM SPECIFICATIONS
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.9rem;">
                <div><span style="color:#94A3B8;">Modulation Scheme:</span> <b style="color:#FFFFFF;">Binary FSK (2-FSK)</b></div>
                <div><span style="color:#94A3B8;">Bit 0 Frequency (f0):</span> <b style="color:#00D9FF;">{st.session_state.f0:.1f} Hz</b></div>
                <div><span style="color:#94A3B8;">Bit 1 Frequency (f1):</span> <b style="color:#00D9FF;">{st.session_state.f1:.1f} Hz</b></div>
                <div><span style="color:#94A3B8;">Transmission Rate:</span> <b style="color:#FFFFFF;">{st.session_state.bit_rate:.1f} bps</b></div>
                <div><span style="color:#94A3B8;">Sampling Frequency:</span> <b style="color:#FFFFFF;">{st.session_state.sampling_rate:.1f} Hz</b></div>
                <div><span style="color:#94A3B8;">Estimated Bandwidth:</span> <b style="color:#10B981;">{st.session_state.bandwidth:.1f} Hz</b></div>
            </div>
        </div>
        """
    )


def render_sensor_data():
    render_html(
        """
        <div style="margin-bottom: 1.2rem;">
            <div style="font-size: 1.85rem; font-weight: 700; color: #FFFFFF;">STAGE 01: SENSOR DATA ACQUISITION</div>
            <div style="font-size: 1.05rem; color: #94A3B8; margin-top: 0.2rem;">
                Acquire real-time environmental IoT telemetry or manually configure simulated sensor values.
            </div>
        </div>
        """
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        render_html(
            """
            <div class="glass-card-3d" style="height: 100%;">
                <div style="font-size: 1.15rem; font-weight: 700; color: #FFFFFF; margin-bottom: 0.75rem;">
                    📝 MANUAL SENSOR INPUT
                </div>
            </div>
            """
        )
        st.number_input("Temperature (°C) [0 - 100]", min_value=0.0, max_value=100.0, step=0.5, key="temperature")
        st.number_input("Humidity (%) [0 - 100]", min_value=0.0, max_value=100.0, step=0.5, key="humidity")

        if st.button("💾 APPLY MANUAL SENSOR VALUES", use_container_width=True, key="btn_apply_sensor"):
            if validate_temperature(st.session_state.temperature) and validate_humidity(st.session_state.humidity):
                st.session_state.reading_mode = "MANUAL"
                st.session_state.timestamp = formatted_timestamp()
                if run_full_simulation():
                    st.success("Manual sensor reading applied and transmitted successfully.")
                    st.rerun()
            else:
                st.warning("Please enter valid temperature and humidity values (0-100).")

    with col2:
        render_html(
            """
            <div class="glass-card-3d" style="height: 100%; display: flex; flex-direction: column; justify-content: space-between;">
                <div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #FFFFFF; margin-bottom: 0.75rem;">
                        🎲 AUTOMATED SIMULATED SENSOR GENERATOR
                    </div>
                    <div style="font-size: 0.95rem; color: #CBD5E1; line-height: 1.6; margin-bottom: 1.2rem;">
                        Generate realistic dynamic environmental readings simulating physical DHT11 / DHT22 temperature and relative humidity sensors.
                    </div>
                </div>
            </div>
            """
        )
        if st.button("⚡ GENERATE RANDOM SENSOR READING", use_container_width=True, key="btn_generate_sensor"):
            temp, hum = generate_sensor_reading()
            st.session_state.temperature = float(temp)
            st.session_state.humidity = float(hum)
            st.session_state.reading_mode = "SIMULATED"
            st.session_state.timestamp = formatted_timestamp()
            if run_full_simulation():
                st.success(f"Simulated sensor reading generated: {temp}°C, {hum}%")
                st.rerun()

    render_html("<div style='height: 18px;'></div>")
    render_html("<div style='font-size: 1.15rem; font-weight: 700; color: #FFFFFF;'>📊 CURRENT SENSOR STATUS (LIVE TELEMETRY)</div>")

    # Render proper sized sensor status grid
    render_sensor_status_grid()

    render_page_nav("HOME", "BINARY CONVERSION", "HOME", "BINARY CONVERSION")


def render_binary_conversion():
    render_html(
        """
        <div style="margin-bottom: 1.2rem;">
            <div style="font-size: 1.85rem; font-weight: 700; color: #FFFFFF;">STAGE 02: BINARY CONVERSION & FRAMING</div>
            <div style="font-size: 1.05rem; color: #94A3B8; margin-top: 0.2rem;">
                Conversion of analog floating-point sensor telemetry into standard 8-bit unsigned digital bytes.
            </div>
        </div>
        """
    )

    if not st.session_state.simulation_ready:
        run_full_simulation()

    temp_bits = format(int(round(st.session_state.temperature)) & 0xFF, "08b")
    hum_bits = format(int(round(st.session_state.humidity)) & 0xFF, "08b")

    col1, col2 = st.columns(2)
    with col1:
        render_html(
            f"""
            <div class="glass-card-3d">
                <div style="font-size: 0.82rem; font-weight: 700; color: #94A3B8; letter-spacing: 0.06em;">BYTE 01: TEMPERATURE PAYLOAD</div>
                <div style="font-size: 1.25rem; font-weight: 700; color: #FFFFFF; margin: 0.35rem 0;">{st.session_state.temperature:.2f} °C ➔ {int(round(st.session_state.temperature))} DEC</div>
                <div class="telemetry-stream-box" style="color: #00D9FF; margin-top: 0.5rem;">
                    {temp_bits}
                </div>
            </div>
            """
        )

    with col2:
        render_html(
            f"""
            <div class="glass-card-3d">
                <div style="font-size: 0.82rem; font-weight: 700; color: #94A3B8; letter-spacing: 0.06em;">BYTE 02: HUMIDITY PAYLOAD</div>
                <div style="font-size: 1.25rem; font-weight: 700; color: #FFFFFF; margin: 0.35rem 0;">{st.session_state.humidity:.2f} % ➔ {int(round(st.session_state.humidity))} DEC</div>
                <div class="telemetry-stream-box" style="color: #10B981; margin-top: 0.5rem;">
                    {hum_bits}
                </div>
            </div>
            """
        )

    render_html("<div style='height: 14px;'></div>")

    render_html(
        f"""
        <div class="glass-card-3d">
            <div style="font-size: 0.85rem; font-weight: 700; color: #00D9FF; letter-spacing: 0.06em; margin-bottom: 0.4rem;">
                COMBINED 16-BIT TELEMETRY FRAME (TRANSMITTED PAYLOAD)
            </div>
            <div class="telemetry-stream-box" style="font-size: clamp(1.2rem, 2.5vw, 1.5rem); letter-spacing: 0.22em;">
                <span style="color: #00D9FF;">{temp_bits}</span> <span style="color: #10B981;">{hum_bits}</span>
            </div>
            <div style="display: flex; justify-content: space-around; font-size: 0.88rem; color: #CBD5E1; margin-top: 0.6rem;">
                <span>Byte 1: Temperature (8 bits)</span>
                <span>Byte 2: Humidity (8 bits)</span>
                <span>Total Frame: 16 Bits</span>
            </div>
        </div>
        """
    )

    render_page_nav("SENSOR DATA", "BFSK MODULATION", "SENSOR DATA", "BFSK MODULATION")


def render_fsk_modulation():
    render_html(
        """
        <div style="margin-bottom: 1.2rem;">
            <div style="font-size: 1.85rem; font-weight: 700; color: #FFFFFF;">STAGE 03: BFSK SIGNAL MODULATION</div>
            <div style="font-size: 1.05rem; color: #94A3B8; margin-top: 0.2rem;">
                Synthesis of continuous phase Binary Frequency Shift Keying waveform representing the digital telemetry stream.
            </div>
        </div>
        """
    )

    if not st.session_state.simulation_ready:
        run_full_simulation()

    signal = st.session_state.analysis_results["signal"]
    time_axis = st.session_state.analysis_results["time_axis"]

    display_samples = min(len(time_axis), 3000)
    fig = plot_bfsk_waveform(
        time_axis[:display_samples],
        signal[:display_samples],
        f"BFSK MODULATED TRANSMISSION WAVEFORM (f0={st.session_state.f0:.0f}Hz, f1={st.session_state.f1:.0f}Hz)",
    )
    st.pyplot(fig)

    render_html("<div style='height: 10px;'></div>")
    cards = st.columns(4)
    params = [
        ("CARRIER FREQUENCY 0 (f0)", f"{st.session_state.f0:.1f} Hz", "Represents Logic '0'"),
        ("CARRIER FREQUENCY 1 (f1)", f"{st.session_state.f1:.1f} Hz", "Represents Logic '1'"),
        ("SYMBOL / BIT RATE", f"{st.session_state.bit_rate:.1f} bps", "1 symbol = 1 bit (BFSK)"),
        ("SAMPLING FREQUENCY", f"{st.session_state.sampling_rate:.1f} Hz", "Nyquist discrete sampling"),
    ]
    for i, (label, value, sub) in enumerate(params):
        with cards[i]:
            render_html(
                f"""
                <div class="glass-card-3d" style="padding: 0.95rem;">
                    <div style="font-size: 0.78rem; font-weight: 700; color: #94A3B8; letter-spacing: 0.06em;">{label}</div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: #FFFFFF; margin: 0.3rem 0 0.1rem 0;">{value}</div>
                    <div style="font-size: 0.75rem; color: #00D9FF;">{sub}</div>
                </div>
                """
            )

    render_page_nav("BINARY CONVERSION", "WIRELESS CHANNEL", "BINARY CONVERSION", "WIRELESS CHANNEL")


def render_wireless_channel():
    render_html(
        """
        <div style="margin-bottom: 1.2rem;">
            <div style="font-size: 1.85rem; font-weight: 700; color: #FFFFFF;">STAGE 04: WIRELESS CHANNEL & AWGN NOISE</div>
            <div style="font-size: 1.05rem; color: #94A3B8; margin-top: 0.2rem;">
                Physical propagation modeling: Additive White Gaussian Noise (AWGN) added to the modulated radio frequency waveform.
            </div>
        </div>
        """
    )

    if not st.session_state.simulation_ready:
        run_full_simulation()

    signal = st.session_state.analysis_results["signal"]
    noisy = st.session_state.noisy_signal
    time_axis = st.session_state.analysis_results["time_axis"]

    display_samples = min(len(signal), 1200)
    fig = plot_noisy_signal(
        time_axis[:display_samples],
        signal[:display_samples],
        noisy[:display_samples],
    )
    st.pyplot(fig)

    render_html("<div style='height: 10px;'></div>")
    metric_cols = st.columns(5)
    values = [
        ("TRANSMIT SIGNAL POWER", f"{st.session_state.signal_power:.4f} W", "Pure BFSK Power"),
        ("NOISE POWER (Pn)", f"{st.session_state.noise_power:.4f} W", "Gaussian Variance"),
        ("CONFIGURED SNR", f"{st.session_state.configured_snr:.2f} dB", "Input Test Ratio"),
        ("ACTUAL CHANNEL SNR", f"{st.session_state.actual_snr:.2f} dB", "Measured Received SNR"),
        ("LINK QUALITY", st.session_state.quality, "Channel Viability"),
    ]
    for idx, (label, value, sub) in enumerate(values):
        with metric_cols[idx]:
            render_html(
                f"""
                <div class="glass-card-3d" style="padding: 0.95rem;">
                    <div style="font-size: 0.74rem; font-weight: 700; color: #94A3B8; letter-spacing: 0.06em;">{label}</div>
                    <div style="font-size: 1.2rem; font-weight: 700; color: #FFFFFF; margin: 0.3rem 0 0.1rem 0;">{value}</div>
                    <div style="font-size: 0.74rem; color: #00D9FF;">{sub}</div>
                </div>
                """
            )

    render_page_nav("BFSK MODULATION", "DEMODULATION", "BFSK MODULATION", "DEMODULATION")


def render_demodulation():
    render_html(
        """
        <div style="margin-bottom: 1.2rem;">
            <div style="font-size: 1.85rem; font-weight: 700; color: #FFFFFF;">STAGE 05: BFSK COHERENT DEMODULATION</div>
            <div style="font-size: 1.05rem; color: #94A3B8; margin-top: 0.2rem;">
                Correlation detection: Received noisy signal is matched against f0 and f1 reference carrier tones to reconstruct digital bits.
            </div>
        </div>
        """
    )

    if not st.session_state.simulation_ready:
        run_full_simulation()

    transmitted = st.session_state.analysis_results["bitstream"]
    recovered = st.session_state.recovered_bits

    col1, col2 = st.columns(2)
    with col1:
        render_html(
            f"""
            <div class="glass-card-3d">
                <div style="font-size: 0.82rem; font-weight: 700; color: #00D9FF; letter-spacing: 0.06em;">ORIGINAL TRANSMITTED BITS (Tx)</div>
                <div class="telemetry-stream-box" style="color: #FFFFFF; margin-top: 0.45rem;">
                    {transmitted}
                </div>
            </div>
            """
        )

    with col2:
        render_html(
            f"""
            <div class="glass-card-3d">
                <div style="font-size: 0.82rem; font-weight: 700; color: #10B981; letter-spacing: 0.06em;">DEMODULATED RECOVERED BITS (Rx)</div>
                <div class="telemetry-stream-box" style="color: #FFFFFF; margin-top: 0.45rem;">
                    {recovered}
                </div>
            </div>
            """
        )

    render_html("<div style='height: 10px;'></div>")
    mismatches = [idx for idx, (tx, rx) in enumerate(zip(transmitted, recovered)) if tx != rx]
    if mismatches:
        st.warning(f"⚠️ {len(mismatches)} Bit Error(s) detected during wireless propagation at indices: {mismatches}")
    else:
        st.success("✅ 100% PERFECT DATA TRANSMISSION - Zero bit errors detected over the wireless channel!")

    # Bit-by-bit correlation verification table
    render_html(
        """
        <div class="glass-card-3d" style="margin-top: 0.8rem; margin-bottom: 0.6rem;">
            <div style="font-size: 1.1rem; font-weight: 700; color: #FFFFFF;">
                🔍 BIT-BY-BIT RECOVERY VERIFICATION (16 BITS)
            </div>
        </div>
        """
    )
    
    # Render bit cards in 2 clean rows of 8 columns each (100% clean, responsive, no code block glitches)
    for row_start in [0, 8]:
        cols = st.columns(8)
        for i in range(8):
            idx = row_start + i
            tx = transmitted[idx]
            rx = recovered[idx]
            is_match = (tx == rx)
            bg = "rgba(16, 185, 129, 0.12)" if is_match else "rgba(239, 68, 68, 0.2)"
            border = "#10B981" if is_match else "#EF4444"
            symbol = "✓ Match" if is_match else "✗ Error"
            with cols[i]:
                render_html(
                    f"""
                    <div style="background: {bg}; border: 1px solid {border}; border-radius: 8px; padding: 0.45rem 0.2rem; text-align: center; margin-bottom: 0.4rem;">
                        <div style="font-size: 0.72rem; color: #94A3B8;">Bit #{idx}</div>
                        <div style="font-size: 0.95rem; font-weight: 700; color: #FFFFFF;">{tx} ➔ {rx}</div>
                        <div style="font-size: 0.72rem; font-weight: 700; color: {border};">{symbol}</div>
                    </div>
                    """
                )

    render_page_nav("WIRELESS CHANNEL", "SIGNAL ANALYSIS", "WIRELESS CHANNEL", "SIGNAL ANALYSIS")


def render_signal_analysis():
    render_html(
        """
        <div style="margin-bottom: 1.2rem;">
            <div style="font-size: 1.85rem; font-weight: 700; color: #FFFFFF;">STAGE 06: MULTI-DOMAIN SIGNAL ANALYSIS</div>
            <div style="font-size: 1.05rem; color: #94A3B8; margin-top: 0.2rem;">
                Comprehensive inspection of waveforms in time domain, digital logic levels, and frequency spectrum (FFT).
            </div>
        </div>
        """
    )

    if not st.session_state.simulation_ready:
        run_full_simulation()

    signal = st.session_state.analysis_results["signal"]
    noised = st.session_state.noisy_signal
    original_bits = st.session_state.analysis_results["bitstream"]
    recovered_bits = st.session_state.recovered_bits
    time_axis = st.session_state.analysis_results["time_axis"]

    col1, col2 = st.columns(2)
    display_samples = min(len(time_axis), 2000)
    with col1:
        st.pyplot(plot_bfsk_waveform(time_axis[:display_samples], signal[:display_samples], "Clean Modulated Waveform s(t)"))
    with col2:
        st.pyplot(plot_noisy_signal(time_axis[:display_samples], signal[:display_samples], noised[:display_samples]))

    st.pyplot(plot_bit_comparison(original_bits, recovered_bits))
    st.pyplot(plot_fft_spectrum(signal, st.session_state.sampling_rate, st.session_state.f0, st.session_state.f1))

    render_page_nav("DEMODULATION", "PERFORMANCE", "DEMODULATION", "PERFORMANCE")


def render_performance():
    render_html(
        """
        <div style="margin-bottom: 1.2rem;">
            <div style="font-size: 1.85rem; font-weight: 700; color: #FFFFFF;">STAGE 07: SYSTEM PERFORMANCE & BER EVALUATION</div>
            <div style="font-size: 1.05rem; color: #94A3B8; margin-top: 0.2rem;">
                Empirical Bit Error Rate (BER) analysis across SNR sweep from 0 dB to 30 dB.
            </div>
        </div>
        """
    )

    if not st.session_state.simulation_ready:
        run_full_simulation()

    fig = plot_ber_curve(st.session_state.SNR_results)
    st.pyplot(fig)

    render_html("<div style='height: 10px;'></div>")
    perf_cols = st.columns(6)
    values = [
        ("BIT ERROR RATE", f"{st.session_state.BER:.5f}", "Measured BER"),
        ("ACTUAL SNR", f"{st.session_state.actual_snr:.2f} dB", "Signal-to-Noise"),
        ("BIT ERRORS", str(st.session_state.bit_errors), "Mismatched Bits"),
        ("TOTAL BITS", str(st.session_state.total_bits), "Telemetry Payload"),
        ("BIT RATE", f"{st.session_state.bit_rate:.1f} bps", "Data Throughput"),
        ("BANDWIDTH", f"{st.session_state.bandwidth:.1f} Hz", "Carson's Approx."),
    ]
    for idx, (label, value, sub) in enumerate(values):
        with perf_cols[idx]:
            render_html(
                f"""
                <div class="glass-card-3d" style="padding: 0.85rem;">
                    <div style="font-size: 0.72rem; font-weight: 700; color: #94A3B8; letter-spacing: 0.06em;">{label}</div>
                    <div style="font-size: 1.2rem; font-weight: 700; color: #FFFFFF; margin: 0.25rem 0 0.1rem 0;">{value}</div>
                    <div style="font-size: 0.72rem; color: #00D9FF;">{sub}</div>
                </div>
                """
            )

    render_html(
        f"""
        <div class="glass-card-3d" style="margin-top: 1.0rem; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.85rem;">
            <div>
                <div style="font-size: 0.82rem; color: #94A3B8; font-weight: 700;">TRANSMISSION QUALITY RATING</div>
                <div style="font-size: 1.3rem; font-weight: 700; color: #00D9FF;">{st.session_state.quality} Communication Link</div>
            </div>
            <div style="font-size: 0.95rem; color: #CBD5E1; max-width: 520px; line-height: 1.5;">
                At current SNR of <b style="color: #FFFFFF;">{st.session_state.actual_snr:.2f} dB</b>, the transmission bit error performance matches standard BFSK theoretical bounds.
            </div>
        </div>
        """
    )

    render_page_nav("SIGNAL ANALYSIS", "FINAL RESULT", "SIGNAL ANALYSIS", "FINAL RESULT")


def render_final_result():
    render_html(
        """
        <div style="margin-bottom: 1.2rem;">
            <div style="font-size: 1.85rem; font-weight: 700; color: #FFFFFF;">STAGE 08: FINAL TELEMETRY RECOVERY REPORT</div>
            <div style="font-size: 1.05rem; color: #94A3B8; margin-top: 0.2rem;">
                End-to-end verification comparing transmitted physical sensor telemetry against received and decoded values.
            </div>
        </div>
        """
    )

    if not st.session_state.simulation_ready:
        run_full_simulation()

    # Success / Failure Banner
    if st.session_state.bit_errors == 0:
        render_html(
            """
            <div class="glass-card-3d" style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(6, 26, 18, 0.95) 100%); border-color: #10B981; padding: 1.2rem; text-align: center; margin-bottom: 1.2rem;">
                <div style="font-size: 1.8rem;">🎉</div>
                <div style="font-size: 1.35rem; font-weight: 700; color: #A7F3D0;">100% RECOVERY SUCCESSFUL</div>
                <div style="font-size: 0.95rem; color: #CBD5E1; margin-top: 0.25rem;">All sensor readings were transmitted, modulated, propagated through noise, and perfectly recovered with zero errors.</div>
            </div>
            """
        )
    else:
        render_html(
            f"""
            <div class="glass-card-3d" style="background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(30, 8, 8, 0.95) 100%); border-color: #EF4444; padding: 1.2rem; text-align: center; margin-bottom: 1.2rem;">
                <div style="font-size: 1.8rem;">⚠️</div>
                <div style="font-size: 1.35rem; font-weight: 700; color: #FECACA;">CHANNEL NOISE DISTORTION DETECTED</div>
                <div style="font-size: 0.95rem; color: #CBD5E1; margin-top: 0.25rem;">{st.session_state.bit_errors} bit error(s) occurred due to low SNR ({st.session_state.actual_snr:.2f} dB). Increase SNR for zero-error recovery.</div>
            </div>
            """
        )

    # Side by side comparison: Transmitted vs Recovered
    col1, col2 = st.columns(2)
    with col1:
        render_html(
            f"""
            <div class="glass-card-3d">
                <div style="font-size: 1.15rem; font-weight: 700; color: #00D9FF; margin-bottom: 0.75rem;">
                    📡 TRANSMITTER (SOURCE TELEMETRY)
                </div>
                <div style="margin-bottom: 0.5rem;"><span style="color:#94A3B8;">Sensor Temperature:</span> <b style="font-size: 1.2rem; color:#FFFFFF;">{st.session_state.temperature:.2f} °C</b></div>
                <div style="margin-bottom: 0.5rem;"><span style="color:#94A3B8;">Sensor Humidity:</span> <b style="font-size: 1.2rem; color:#FFFFFF;">{st.session_state.humidity:.2f} %</b></div>
                <div style="margin-bottom: 0.4rem; color:#94A3B8; font-size:0.9rem;">Transmitted Binary Frame:</div>
                <div class="telemetry-stream-box" style="color: #00D9FF;">
                    {st.session_state.binary_data}
                </div>
            </div>
            """
        )

    with col2:
        render_html(
            f"""
            <div class="glass-card-3d">
                <div style="font-size: 1.15rem; font-weight: 700; color: #10B981; margin-bottom: 0.75rem;">
                    🎯 RECEIVER (RECOVERED TELEMETRY)
                </div>
                <div style="margin-bottom: 0.5rem;"><span style="color:#94A3B8;">Decoded Temperature:</span> <b style="font-size: 1.2rem; color:#FFFFFF;">{st.session_state.recovered_temperature} °C</b></div>
                <div style="margin-bottom: 0.5rem;"><span style="color:#94A3B8;">Decoded Humidity:</span> <b style="font-size: 1.2rem; color:#FFFFFF;">{st.session_state.recovered_humidity} %</b></div>
                <div style="margin-bottom: 0.4rem; color:#94A3B8; font-size:0.9rem;">Demodulated Binary Frame:</div>
                <div class="telemetry-stream-box" style="color: #10B981;">
                    {st.session_state.recovered_bits}
                </div>
            </div>
            """
        )

    render_html("<div style='height: 18px;'></div>")

    # 3D Action Controls
    col_b1, col_b2, col_b3 = st.columns(3)
    with col_b1:
        if st.button("← BACK TO PERFORMANCE", use_container_width=True, key="fin_btn_back"):
            st.session_state.page = "PERFORMANCE"
            st.rerun()
    with col_b2:
        if st.button("⚡ SIMULATE NEW SENSOR RUN", use_container_width=True, key="fin_btn_new"):
            temp, hum = generate_sensor_reading()
            st.session_state.temperature = float(temp)
            st.session_state.humidity = float(hum)
            st.session_state.reading_mode = "SIMULATED"
            run_full_simulation()
            st.rerun()
    with col_b3:
        if st.button("🏠 RETURN TO HOME", use_container_width=True, key="fin_btn_home"):
            st.session_state.page = "HOME"
            st.rerun()


def page_router():
    if st.session_state.page == "HOME":
        render_home()
    elif st.session_state.page == "SENSOR DATA":
        render_sensor_data()
    elif st.session_state.page == "BINARY CONVERSION":
        render_binary_conversion()
    elif st.session_state.page == "BFSK MODULATION":
        render_fsk_modulation()
    elif st.session_state.page == "WIRELESS CHANNEL":
        render_wireless_channel()
    elif st.session_state.page == "DEMODULATION":
        render_demodulation()
    elif st.session_state.page == "SIGNAL ANALYSIS":
        render_signal_analysis()
    elif st.session_state.page == "PERFORMANCE":
        render_performance()
    elif st.session_state.page == "FINAL RESULT":
        render_final_result()


def main():
    init_session_state()
    inject_custom_css()

    if time.time() < st.session_state.splash_time and "splash_seen" not in st.session_state:
        show_splash_screen()
        st.session_state.splash_seen = True
        st.rerun()

    render_sidebar()
    page_router()


if __name__ == "__main__":
    main()
