import json
import os
import time
from typing import Any, Dict, List

import pandas as pd
import mlflow
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from databricks_langchain import ChatDatabricks


# -----------------------------
# Load JSONL (string or file path)
# -----------------------------
def _load_jsonl(jsonl: str) -> pd.DataFrame:
    if os.path.exists(jsonl):
        with open(jsonl, "r", encoding="utf-8") as f:
            content = f.read()
    else:
        content = jsonl

    rows = [json.loads(line) for line in content.splitlines() if line.strip()]
    return pd.DataFrame(rows)


# -----------------------------
# Extract tool calls from graph messages
# -----------------------------
def _extract_tool_calls(messages: List[Any]) -> List[str]:
    tools = []
    for m in messages:
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                if name:
                    tools.append(name)
    return tools


# -----------------------------
# LLM Judge
# -----------------------------
JUDGE_SYSTEM = SystemMessage(
    content=(
        "You are grading a travel assistant answer.\n"
        "Return strict JSON only:\n"
        "{\"score\": 0-5}\n"
        "5=Fully correct. 4=Mostly correct. 3=Partially correct. "
        "2=Major issues. 1=Wrong. 0=Unsafe or fabricated."
    )
)


def _judge(judge_llm, question: str, ground_truth: str, prediction: str) -> int:
    prompt = (
        f"QUESTION:\n{question}\n\n"
        f"GROUND_TRUTH:\n{ground_truth}\n\n"
        f"MODEL_ANSWER:\n{prediction}\n"
    )
    resp = judge_llm.invoke([JUDGE_SYSTEM, HumanMessage(content=prompt)])

    try:
        return int(json.loads(resp.content).get("score", 0))
    except Exception:
        return 0


# -----------------------------
# MAIN RUNNER (MINIMAL)
# -----------------------------
def run_mlflow_eval(graph, eval_jsonl: str, judge_endpoint: str):
    df = _load_jsonl(eval_jsonl)
    judge_llm = ChatDatabricks(endpoint=judge_endpoint)

    results = []
    mlflow.set_experiment("/Users/sonalivedaraj@gmail.com/batch-experiment")
    with mlflow.start_run(run_name="airline_agent_eval"):

        for row in df.to_dict("records"):
            q = row["question"]
            gt = row["expected_answer"]

            with mlflow.start_run(nested=True):

                start = time.time()
                out = graph.invoke({"messages": [HumanMessage(content=q)]})
                latency = int((time.time() - start) * 1000)

                messages = out.get("messages", [])
                final_answer = messages[-1].content if messages else ""

                tools = _extract_tool_calls(messages)
                judge_score = _judge(judge_llm, q, gt, final_answer)

                # Log minimal per-case metrics
                mlflow.log_metric("latency_ms", latency)
                mlflow.log_metric("num_tool_calls", len(tools))
                mlflow.log_metric("judge_score", judge_score)
                mlflow.log_param("tool_flow", " → ".join(tools))

                results.append({
                    "question": q,
                    "tool_flow": tools,
                    "judge_score": judge_score,
                    "latency_ms": latency
                })

        results_df = pd.DataFrame(results)

        # Log minimal aggregate metrics
        mlflow.log_metric("avg_judge_score", results_df["judge_score"].mean())
        mlflow.log_metric("avg_latency_ms", results_df["latency_ms"].mean())
        mlflow.log_metric("avg_tool_calls", results_df["tool_flow"].apply(len).mean())

    return results_df