# src/italian_numbers.py

from __future__ import annotations

import re
from typing import Optional


UNITS = {
    "zero": 0,
    "uno": 1,
    "un": 1,
    "una": 1,
    "due": 2,
    "tre": 3,
    "quattro": 4,
    "cinque": 5,
    "sei": 6,
    "sette": 7,
    "otto": 8,
    "nove": 9,
}

TEENS = {
    "dieci": 10,
    "undici": 11,
    "dodici": 12,
    "tredici": 13,
    "quattordici": 14,
    "quindici": 15,
    "sedici": 16,
    "diciassette": 17,
    "diciotto": 18,
    "diciannove": 19,
}

TENS = {
    "venti": 20,
    "trenta": 30,
    "quaranta": 40,
    "cinquanta": 50,
    "sessanta": 60,
    "settanta": 70,
    "ottanta": 80,
    "novanta": 90,
}

HUNDREDS = {
    "cento": 100,
    "duecento": 200,
}


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("é", "e").replace("è", "e")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def italian_word_to_number(text: str) -> Optional[int]:
    """
    Supports examples like:
    - settantadue -> 72
    - ottanta -> 80
    - novantasette -> 97
    - cento -> 100
    - centotrenta -> 130
    - centoventi -> 120
    - centotrentaquattro -> 134
    """
    t = _normalize(text).replace(" ", "")
    if not t:
        return None

    if t.isdigit():
        return int(t)

    if t in UNITS:
        return UNITS[t]

    if t in TEENS:
        return TEENS[t]

    if t in TENS:
        return TENS[t]

    if t in HUNDREDS:
        return HUNDREDS[t]

    # 21-99 e.g. settantadue, novantasette
    for tens_word, tens_value in TENS.items():
        if t.startswith(tens_word):
            remainder = t[len(tens_word):]
            if remainder == "":
                return tens_value
            if remainder in UNITS:
                return tens_value + UNITS[remainder]

    # 100-199 e.g. centotrenta, centotrentaquattro
    if t.startswith("cento"):
        remainder = t[len("cento"):]
        if remainder == "":
            return 100
        sub = italian_word_to_number(remainder)
        if sub is not None:
            return 100 + sub

    # 200
    if t == "duecento":
        return 200

    return None


def extract_number_from_text(text: str, min_value: int | None = None, max_value: int | None = None) -> Optional[int]:
    """
    Finds first number either in digits or Italian words.
    """
    normalized = _normalize(text)

    # First try digits
    m = re.search(r"\b\d{1,3}\b", normalized)
    if m:
        value = int(m.group(0))
        if (min_value is None or value >= min_value) and (max_value is None or value <= max_value):
            return value

    # Then try word tokens and joined neighbors
    words = normalized.split()

    # single token
    for word in words:
        value = italian_word_to_number(word)
        if value is not None:
            if (min_value is None or value >= min_value) and (max_value is None or value <= max_value):
                return value

    # joined 2-token combinations, e.g. "cento trenta"
    for i in range(len(words) - 1):
        combo = words[i] + words[i + 1]
        value = italian_word_to_number(combo)
        if value is not None:
            if (min_value is None or value >= min_value) and (max_value is None or value <= max_value):
                return value

    return None