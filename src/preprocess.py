import re


def preprocess_text(text: str) -> str:
    if not text:
        return ""

    t = text.strip()

    wrapper_patterns = [
        r"^Ecco la nota clinica domiciliare ADI in italiano, resa più naturale e professionale:\s*",
        r"^Ecco la nota clinica domiciliare ADI in italiano:\s*",
        r"^Ecco la nota clinica domiciliare in italiano:\s*",
        r"^Ecco la versione rivista della nota clinica in italiano:\s*",
        r"^Ecco la nota clinica domiciliare:\s*",
        r"^Nota clinica domiciliare ADI:\s*",
        r"^Nota clinica domiciliare:\s*",
    ]
    for p in wrapper_patterns:
        t = re.sub(p, "", t, flags=re.IGNORECASE)

    t = re.split(r"\bNota:\b", t, flags=re.IGNORECASE)[0]
    t = re.sub(r"\s+", " ", t).strip()

    return t


def preprocess(text: str) -> str:
    return preprocess_text(text)


def clean_text(text: str) -> str:
    return preprocess_text(text)


def clean(text: str) -> str:
    return preprocess_text(text)


def normalize_text(text: str) -> str:
    return preprocess_text(text)


def prepare_text(text: str) -> str:
    return preprocess_text(text)