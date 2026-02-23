# Databricks notebook source
# MAGIC %pip install -U langchain langchain-core langchain-community langgraph databricks-langchain databricks-ai-bridge
# MAGIC %pip install faiss-cpu
# MAGIC dbutils.library.restartPython()
# MAGIC

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC %reload_ext autoreload

# COMMAND ----------

from langchain_core.messages import HumanMessage
from src.agent import build_agent
from src.utils import run_agent_loop, run_agent_loop_with_mlflow

agent_app = build_agent()

# COMMAND ----------

run_agent_loop(agent_app)


# COMMAND ----------

# DBTITLE 1,chat with logging
run_agent_loop_with_mlflow(agent_app)

# COMMAND ----------

# DBTITLE 1,Test Retrieval
from src.retrieval import PolicyRetriever
from src.helpers import load_text_file

policy_text = load_text_file("data/policies.md")
retriever = PolicyRetriever(policy_text=policy_text, k=3)

questions = [
    "If my spirit airlines is delayed overnight due to the airline’s fault, will they provide hotel and meal vouchers?",
    "What specific services (food, water, lavatories, medical attention) must an airline provide during a tarmac delay?",
    "Within how many days must an airline issue a refund to a credit card after a ticket cancellation?",
    "What is the liability limit for lost, damaged, or delayed baggage on domestic flights?",
    "What written notice must an airline provide to a passenger who is involuntarily bumped?",
    "If I am denied boarding because the flight was overbooked, what compensation am I entitled to?",
    "Where and how can a passenger file a formal complaint with the U.S. Department of Transportation?",
    "Was NK1200 delayed on 2023-12-24"
]

for q in questions:
    results = retriever.search_with_scores(q, k=3)
    print("=" * 80)
    print("Query:", q)
    for i, (doc, score) in enumerate(results, start=1):
        path = (doc.metadata or {}).get("path", "Unknown Section")
        print(f"{i}. score={score:.4f} | {path}")


# COMMAND ----------

# DBTITLE 1,batch test
from src.agent import build_agent
agent_app = build_agent()

from src.test.run_mlflow_eval import run_mlflow_eval

results_df = run_mlflow_eval(
    graph=agent_app,
    eval_jsonl="src/test/testcases.jsonl",
    judge_endpoint="databricks-gpt-5-nano"
)

display(results_df)

# COMMAND ----------

displayHTML(f"<pre class='mermaid'>{agent_app.get_graph().draw_mermaid()}</pre>")
displayHTML("""<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<script>mermaid.initialize({startOnLoad:true});</script>""")


# COMMAND ----------

# why was flight UA0351 delayed on 2023-12-24 ?
# Was NK1200 delayed on 2023-12-24?
# can you give list of all flights from lax to dfw ? on 2023-12-24?