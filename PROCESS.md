## A) Architecture (Design Approach):

### High‑Level Flow
![graph](./graph.png)

### Design Approach
Planner–Executor–Synthesizer Pattern (Iterative Tool-Calling Agent)

This is a planner-driven agent architecture built using LangGraph. The Planner decides whether to call a tool, ask a clarification question, or stop. Tool calls are executed in a separate node, and the loop continues until the Planner decides enough information has been gathered. Then a Synthesizer generates the final structured response strictly from tool outputs.
- The **planner** is the central controller. It decides whether to call a tool or finish the task, looping until all required information is gathered.
- All external data access is isolated into dedicated **tools** (`flight_query`, `policy_search`) so retrieval logic stays deterministic, testable, and separate from reasoning.
- Tools return minimal, structured JSON. This keeps the LLM focused on synthesis instead of raw data handling and prevents hallucination.
- The **planner–tool loop** allows multi‑step queries (e.g., flight status + policy eligibility) without mixing responsibilities.
- The **synthesizer** produces the final, friendly answer using only tool outputs, ensuring clarity and safety.
- Conversation‑aware routing (last N user messages) makes follow‑ups natural while keeping the planner prompt small.

I separated synthesis from planning to ensure clean responsibility boundaries. The planner focuses on orchestration logic, while the synthesizer focuses purely on generating a coherent, well-structured response grounded in tool outputs

This matches the “agentic workflow” requirement and keeps the orchestrator readable and testable.

#### Data sources + tool separation
I separated the two data sources into two tools with clear responsibilities:

1) **Policy RAG Tool**
- `retrieval.py` builds an **in-memory FAISS** vector index once per notebook session and then supports similarity search.
- Clean the markdown (remove noise)
- Split it into meaningful chunks (preserve section structure)
- Embed the chunks (Databricks embeddings endpoint)
- Store embeddings in FAISS (in-memory vector store)
- Provide search() methods so your policy tool can retrieve relevant snippets

2) **Flight Ops Tool**
- Queries `hive_metastore.default.ontime_cleaned` via Spark SQL to answer status, delay, timing, and route questions, using validated filters (flight/date/origin/dest) to prevent full-table scans.
- Used ontime_dictionary.txt during development to understand the schema and define safe column bundles (CORE + intent-based groups).
- Uses deterministic (keyword-based) intent detection and falls back to minimal CORE columns when needed to ensure robustness and avoid schema hallucination.

---

## B) Trade-offs (Demo vs Production)

#### 1. Vector store choice
**Demo:** FAISS (in-memory, zero infra, fastest to iterate in a notebook).  
**Production:** Databricks Vector Search (managed, scalable, UC governance, easier ops).

#### 2. SQL construction / security
**Demo:** SQL queries are built dynamically but only using predefined columns and filters that I manually validated. The allowed fields and conditions are somewhat controlled in code (hardcoded bundles).                
**Production:** Store the data dictionary in Unity Catalog as a governed Delta table and use it as a metadata-driven control layer to enforce allowlists, validate inputs, securely generate SQL, enable auditing, and enforce role-based access policies.

#### 3. Response Quality & Hallucination Control
**Demo:** Return the first LLM answer using top-k retrieved chunks, with minimal validation.                               
**Production** Add an answer validation loop (Self-RAG / Corrective RAG) that checks if the response is grounded in retrieved sources, triggers re-retrieval or query refinement when confidence is low, and enforces “cite or abstain” behavior to reduce hallucinations.

#### 4. Conversation memory
**Demo:** Keep last N messages in RAM.  
**Production:** Store long-term memory in an external persistent store and extract key facts or summaries from conversations into structured records. Retrieve relevant past context only when needed, and enforce retention, expiration, and privacy policies to manage cost and compliance.


#### 5. Scalability & Deployment
**Demo:** Run on a single interactive Databricks notebook/cluster with local state.                                
**Production:** Deploy model using Databricks Model Serving with built-in autoscaling for inference, while using autoscaling SQL Warehouses and Jobs clusters for data processing workloads.

---
## C) Retrospective (Hardest Parts)

#### 1) Correct Tool Routing & Over-Invocation

One of the most challenging aspects was ensuring the agent invoked the correct tool at the right time. Initially, the model occasionally called the flight tool when the user only needed policy guidance (and vice versa).

Addressing this required iterative refinement of:

- Tool descriptions  
- System prompt constraints  
- Output structure (to avoid over-retrieval)

The final design enforces clearer routing behavior while still allowing multi-step flows when necessary (e.g., flight lookup → policy entitlement explanation).

---

#### 2) Notebook Stability & Runtime Issues

Running the agent interactively inside Databricks notebooks introduced instability, particularly around threaded input handling and streaming logic.

The notebook kernel crashed intermittently, and diagnosing the issue required isolating:

- Threaded input mechanisms  
- Event streaming behavior  
- Spark query latency interactions

Through step-by-step testing, I identified that the instability was primarily caused by the interaction between asynchronous input logic and the notebook runtime under load.

The system was stabilized by carefully testing components independently and constraining tool output size to reduce memory and execution pressure.

---

#### 3) Dynamic SQL & Schema Strategy

Initially, I attempted a fully dynamic column-selection strategy by loading `ontime_dictionary.txt` into memory and selecting columns based on user keywords.

While flexible, this approach:

- Became increasingly complex to maintain  
- Struggled with ambiguous question patterns  
- Increased the risk of invalid or hallucinated SQL  
- Reduced predictability of query behavior

After experimentation and testing, I intentionally shifted to predefined intent-based column bundles (CORE + grouped fields).

This improved:

- Query determinism  
- Debuggability  
- Schema safety  
- Performance control

The final approach balances flexibility with robustness.

---

#### 4) Token Growth & Context Management

As conversations grew longer, passing the full message history increased latency and occasionally impacted routing consistency.

To mitigate this:

- I limited conversation memory to a rolling window  
- I constrained tool outputs to be compact  
- I identified historical summarization as a production-ready enhancement

This ensured stable performance while maintaining sufficient conversational context.


