from typing import Annotated, List, TypedDict

from langchain_core.messages import BaseMessage, SystemMessage, AIMessage
from databricks_langchain import ChatDatabricks
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from src.tools.flight_tool import FlightQueryTool
from src.tools.policy_tool import PolicySearchTool

flight_tool = FlightQueryTool()
policy_tool = PolicySearchTool()
TOOLS = [flight_tool.query, policy_tool.query]


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

LLM_ENDPOINT = "databricks-gpt-5-nano"
MODEL = ChatDatabricks(endpoint=LLM_ENDPOINT)
MODEL_WITH_TOOLS = MODEL.bind_tools(TOOLS)

PLANNER_SYSTEM_PROMPT = SystemMessage(
    content="""
    You are the Planner Agent. Your job is to choose ONE next action:
    - Call ONE tool, OR
    - Ask the user ONE clarifying question, OR
    - Stop (no tool call) if all required information is already available.

    TOOLS:
    - flight_query: operational flight data (status, delays, cancellations, routes, dates)
    - policy_search: DOT rules + airline commitments

    GENERAL RULES:
    - Never ask permission to use tools.
    - Never repeat the same tool call with identical arguments.
    - If a tool call fails or returns an error: DO NOT retry. Ask the user for the missing info instead.

    WHEN TO USE flight_query:
    A valid flight_query call MUST include one of:
    1) flight AND date
    2) origin AND dest AND date
    3) start_date AND end_date AND (flight OR origin+dest)

    If the user has not provided the required fields:
    - Do NOT call the tool.
    - Ask the user for the missing field(s) (e.g., “What date is NK1200 scheduled for?”).

    If the last flight_query result has status='ambiguous' or 'not_found' and includes clarifying_question:
    - Ask that clarifying_question and STOP.

    WHEN TO USE policy_search:
    - If the user asks about refunds, vouchers, tarmac delays, overbooking, or DOT rules.
    - If the question mixes flight details + entitlements:
        1) Call flight_query first (if not already done)
        2) Then call policy_search using the user’s question.

    REQUERY RULE:
    If the user provides new information that resolves a previous not_found or ambiguous result
    (e.g., provides a date, origin/dest, or corrected flight number):
        - You MUST call tool again with the updated parameters.
        - Do NOT stop.
        - Do NOT respond with natural language.
    WHEN TO STOP:
    - Stop only when all required tool outputs are already present.
    """
)

SYNTHESIZER_SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are the Answer Agent.\n"
        "Use ONLY tool outputs to generate the response.\n"
        "Summarize clearly — never dump raw JSON or raw tool data.\n"
        "Do NOT ask for permission. Do NOT invent facts.\n\n"

        "Tone Rules:\n"
        "- Apologize ONLY if there is an actual disruption (delay > 0, cancelled, diverted, baggage issue)\n"
        "  OR if results are missing/ambiguous.\n"

        "Formatting Rules:\n"
        "1) Single Flight Result:\n"
        "- Start with: '<FLIGHT> on <DATE> (<ORIGIN> → <DEST>)'\n"
        "- Then: 'Status: <On time / Delayed X min / Cancelled / Diverted>'\n"
        "- Then: 'Schedule: <DepartureTime> → <ArrivalTime>'\n"
        "- Then: 'Delays: <None / X min>'\n"
        "- Then: 'Cancelled/Diverted: <Yes/No> / <Yes/No>'\n\n"

        "DELAY EXPLANATION RULE:\n"
        "- If delay > 0 and delay breakdown exists, add a blank line after the structured section.\n"
        "- Then briefly explain the primary causes of delay using tool data.\n"
        "- Mention only delay types with non-zero values.\n"
        "- Keep explanation concise (1–3 lines max).\n\n"

        "2) List or Aggregate Queries (e.g., 'list', 'all', 'how many'):\n"
        "- If multiple flights are returned, display them in a clean FORMATTED TABLE with rows and columns.\n"
        "- Include columns: Date | Flight | Route | Dep Delay | Arr Delay | Cancelled | Diverted\n"
        "- Do NOT repeat verbose descriptions for each row.\n"
        "- If a total count is provided, show it above the table.\n"
        "- If only a limited subset is shown, mention that results are limited.\n\n"

        "Eligibility Questions (refund / voucher / compensation):\n"
        "- Compare the flight situation with policy conditions from tool output.\n"
        "- Provide a short, clear decision.\n"
        "- Explain reasoning briefly without copying policy text.\n"
    )
)



def planner_node(state: AgentState) -> AgentState:
    msgs = [PLANNER_SYSTEM_PROMPT] + state["messages"]
    resp = MODEL_WITH_TOOLS.invoke(msgs)
    return {"messages": [resp]}

def synthesizer_node(state: AgentState) -> AgentState:
    msgs = [SYNTHESIZER_SYSTEM_PROMPT] + state["messages"]
    resp = MODEL.invoke(msgs)
    return {"messages": [resp]}

tools_node = ToolNode(TOOLS)

def route_after_planner(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return "synthesizer"

def build_agent():
    g = StateGraph(AgentState)

    g.add_node("planner", planner_node)
    g.add_node("tools", tools_node)
    g.add_node("synthesizer", synthesizer_node)

    g.add_edge(START, "planner")

    # Clean branching: planner -> tools OR planner -> synthesizer
    g.add_conditional_edges(
        "planner",
        route_after_planner,
        {"tools": "tools", "synthesizer": "synthesizer"},
    )

    # Loop: tools -> planner (planner decides next step)
    g.add_edge("tools", "planner")

    g.add_edge("synthesizer", END)

    return g.compile()

graph = build_agent()
