import re
from typing import List, Any, Optional, Dict
from langchain_core.documents import Document


def load_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def parse_flight(flight: str) -> Dict[str, str]:
    """
    UA123 -> {"airline": "UA", "flight_number": "123"}
    """
    m = re.match(r"^\s*([A-Za-z]{2,3})\s*0*([0-9]{1,4})\s*$", flight)
    if not m:
        raise ValueError("Flight must look like 'UA123' (airline code + number).")
    return {"airline": m.group(1).upper(), "flight_number": m.group(2)}


def format_date(date: str) -> str:
    """
    Accepts only ISO date format: YYYY-MM-DD.
    Returns the same string if valid.
    """
    d = date.strip()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", d):
        raise ValueError("Date must be in YYYY-MM-DD format (e.g. 2023-12-29)")
    return d

def format_time(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).zfill(4)
    if len(s) != 4 or not s.isdigit():
        return str(x)
    return f"{s[0:2]}:{s[2:4]}"
