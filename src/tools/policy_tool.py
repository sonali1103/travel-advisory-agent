from typing import Optional, List
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from src.retrieval import PolicyRetriever
from src.helpers import load_text_file

AIRLINE_NAME_FROM_CODE = {
    "AS": "Alaska Airlines",
    "G4": "Allegiant Air",
    "AA": "American Airlines",
    "DL": "Delta Air Lines",
    "F9": "Frontier Airlines",
    "HA": "Hawaiian Airlines",
    "B6": "JetBlue Airways",
    "WN": "Southwest Airlines",
    "NK": "Spirit Airlines",
    "UA": "United Airlines",
}


def normalize_airline(airline: Optional[str]):
    """
    Accepts airline code (UA) or airline name.
    Returns (code, name)
    """
    if not airline:
        return None, None

    airline = airline.strip().upper()
    
    if len(airline) == 2:
        return airline, AIRLINE_NAME_FROM_CODE.get(airline)

    for code, name in AIRLINE_NAME_FROM_CODE.items():
        if airline == name.upper():
            return code, name

    return None, airline  # Unknown airline, return as-is


# -----------------------------
# Input Schema
# -----------------------------
class PolicySearchInput(BaseModel):
    query: str = Field(..., description="Natural language policy question")
    airline: Optional[str] = Field(
        None,
        description="Optional airline code like UA, AA, DL or airline name like United Airlines, Alaska Airlines"
    )


# -----------------------------
# Output Schema
# -----------------------------
class PolicySearchOutput(BaseModel):
    ok: bool
    query: str
    match_count: int
    matches: List[dict]
    airline_code: Optional[str] = None
    airline_name: Optional[str] = None


# -----------------------------
# Tool Implementation
# -----------------------------
class PolicySearchTool:

    def __init__(self, policies_path: str = "data/policies.md", k: int = 3):
        policies_text = load_text_file(policies_path)
        self.retriever = PolicyRetriever(policy_text=policies_text, k=k)
        self.k = k

        self.query = tool(
            "policy_search",
            args_schema=PolicySearchInput
        )(self._query)

    def _query(self, query: str, airline: Optional[str] = None):
        """
        Search airline policies for passenger entitlements.

        Use this tool for ANY question about:
        - vouchers
        - meals
        - refunds
        - cancellations
        - rebooking rights
        - DOT rules
        - overbooking
        - tarmac delays

        Inputs:
            query: natural language policy question
            airline: optional airline code or airline name

        Output:
            Structured list of top matching policy snippets.
        """
        q = (query or "").strip()
        if not q:
            return PolicySearchOutput(
                ok=False,
                query="",
                match_count=0,
                matches=[]
            ).model_dump()

        # Normalize airline
        code, name = normalize_airline(airline)

        if name:
            q = f"{name} policy commitments: {q}"
        elif airline:
            q = f"{airline} policy commitments: {q}"

        docs = self.retriever.search(q, k=self.k)

        matches = []
        for d in docs:
            matches.append({
                "text": d.page_content,
                "path": (d.metadata or {}).get("path"),
                "chunk": (d.metadata or {}).get("chunk"),
            })

        return PolicySearchOutput(
            ok=True,
            query=q,
            match_count=len(matches),
            matches=matches,
            airline_code=code,
            airline_name=name,
        ).model_dump()
