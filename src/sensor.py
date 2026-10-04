from __future__ import annotations

import random
from datetime import datetime


MIN_TEMP = 0.0
MAX_TEMP = 100.0
MIN_HUM = 0.0
MAX_HUM = 100.0


def validate_temperature(value: float) -> bool:
    return MIN_TEMP <= float(value) <= MAX_TEMP


def validate_humidity(value: float) -> bool:
    return MIN_HUM <= float(value) <= MAX_HUM


def generate_sensor_reading() -> tuple[float, float]:
    temperature = round(random.uniform(20.0, 40.0), 2)
    humidity = round(random.uniform(40.0, 90.0), 2)
    return temperature, humidity


def formatted_timestamp() -> str:
    return datetime.now().strftime("%d %b %Y, %H:%M:%S")
