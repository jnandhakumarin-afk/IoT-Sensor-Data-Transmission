from __future__ import annotations


def int_to_8bit(value: int | float) -> str:
    value_int = int(round(float(value)))
    return format(value_int & 0xFF, "08b")


def temperature_to_bits(value: float) -> str:
    value_int = int(round(float(value)))
    if not 0 <= value_int <= 255:
        raise ValueError("Temperature must map to a valid 8-bit unsigned value.")
    return format(value_int, "08b")


def humidity_to_bits(value: float) -> str:
    value_int = int(round(float(value)))
    if not 0 <= value_int <= 255:
        raise ValueError("Humidity must map to a valid 8-bit unsigned value.")
    return format(value_int, "08b")


def combine_sensor_bits(temperature: float, humidity: float) -> str:
    return f"{temperature_to_bits(temperature)}{humidity_to_bits(humidity)}"


def bits_to_integer(bits: str) -> int:
    return int(bits, 2)


def recover_sensor_values(binary_stream: str) -> tuple[int, int]:
    if len(binary_stream) < 16:
        raise ValueError("Binary stream is too short to recover sensor values.")
    temp_bits = binary_stream[:8]
    humidity_bits = binary_stream[8:16]
    return bits_to_integer(temp_bits), bits_to_integer(humidity_bits)
