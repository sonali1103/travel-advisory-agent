from langchain_core.tools import tool
from pydantic import BaseModel, Field, model_validator
from typing import Optional, List
from databricks.sdk.runtime import spark
from src.helpers import parse_flight, format_date


TABLE = "hive_metastore.default.ontime_cleaned"


# -----------------------------
# Input Schema
# -----------------------------
class FlightQueryInput(BaseModel):
    question: str = Field(..., description="Exact user message")
    flight: Optional[str] = Field(None, description="e.g., AA205, UA2726")
    date: Optional[str] = Field(None, description="YYYY-MM-DD")
    origin: Optional[str] = None
    dest: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    @model_validator(mode="after")
    def validate(self):
        # Require at least one usable query pattern:
        has_single = self.flight and self.date
        has_route_day = self.origin and self.dest and self.date
        has_range = (self.start_date and self.end_date) and (self.flight or (self.origin and self.dest))

        if not (has_single or has_route_day or has_range):
            raise ValueError("Provide flight+date OR origin+dest+date OR (start_date+end_date plus flight/route).")
        return self


# -----------------------------
# Output Schema
# -----------------------------
class FlightQueryOutput(BaseModel):
    status: str
    question: str
    clarifying_question: Optional[str] = None
    sql: str
    row_count: int
    rows: List[dict]

class FlightQueryTool:

    def __init__(self):
        self.query = tool(
            "flight_query",
            args_schema=FlightQueryInput
        )(self._query)

    def _query(self, question, flight=None, date=None,
               start_date=None, end_date=None, origin=None, dest=None):
        """
        Query flight operations data.

        Use this tool for ANY question about:
        - flight status
        - delays
        - cancellations
        - origin/destination routes
        - how many flights (aggregates)
        - date or date-range queries
        """

        q = question.lower()
        is_aggregate = any(k in q for k in ["how many", "count", "total", "list", "all"])
        LIMIT = 50 if is_aggregate else 5

        where = []
        if flight:
            parsed = parse_flight(flight)
            airline = parsed["airline"]
            number = parsed["flight_number"]
            where.append(f"Reporting_Airline = '{airline}'")
            where.append(f"Flight_Number_Reporting_Airline = '{number}'")

        if date:
            where.append(f"FlightDate = '{format_date(date)}'")

        if start_date and end_date:
            where.append(
                f"FlightDate BETWEEN '{format_date(start_date)}' AND '{format_date(end_date)}'"
            )

        if origin:
            where.append(f"Origin = '{origin}'")

        if dest:
            where.append(f"Dest = '{dest}'")
        where_sql = " AND ".join(where) if where else "1=1"

        # -----------------------------
        # Build and Execute SQL
        # -----------------------------
        sql = f"""
            SELECT
                FlightDate,
                Reporting_Airline,
                Flight_Number_Reporting_Airline,
                Origin,
                Dest,
                DepDelayMinutes,
                ArrDelayMinutes,
                DepTime,
                ArrTime,
                Cancelled,
                Diverted,
                WeatherDelay,
                NASDelay,
                CarrierDelay,
                SecurityDelay,
                LateAircraftDelay
            FROM {TABLE}
            WHERE {where_sql}
            LIMIT {LIMIT}
        """
        df = spark.sql(sql)
        rows = [r.asDict(recursive=True) for r in df.collect()]

        # -----------------------------
        # Build Output
        # -----------------------------
        if len(rows) == 0:
            return FlightQueryOutput(
                status="not_found",
                clarifying_question=(
                    "I couldn't find any rows for that query. "
                    "Can you confirm the flight (e.g., OO3400) and date (YYYY-MM-DD), "
                    "or provide origin/destination airport codes (3 letters like SEA)?"
                ),
                question=question,
                sql=sql,
                row_count=0,
                rows=[],
            ).model_dump()

        if not is_aggregate and len(rows) > 1:
            return FlightQueryOutput(
                status="ambiguous",
                clarifying_question=(
                    "I found multiple matching flight records. "
                    "Which one do you mean?"
                ),
                question=question,
                sql=sql,
                row_count=len(rows),
                rows=rows
            ).model_dump()

        return FlightQueryOutput(
            status="ok",
            question=question,
            sql=sql,
            row_count=len(rows),
            rows=rows
        ).model_dump()
