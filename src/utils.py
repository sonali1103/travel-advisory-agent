import threading
from langchain_core.messages import HumanMessage, AIMessage
import mlflow
import json
import time

def get_input_with_timeout(prompt: str, timeout: int):
    """Get user input with a timeout using threading."""
    user_input = [None]
    input_received = threading.Event()

    def input_thread():
        try:
            user_input[0] = input(prompt)
            input_received.set()
        except Exception:
            pass

    thread = threading.Thread(target=input_thread, daemon=True)
    thread.start()

    if input_received.wait(timeout):
        return user_input[0]
    return None

def run_agent_loop(agent_app, input_timeout: int = 120, max_memory_messages: int = 20):
    print("--- Passenger Advocate Agent ---")
    print("Type 'quit' to exit.\n")

    messages = []

    while True:
        #user_input = get_input_with_timeout("User: ", input_timeout)

        user_input = input("User: ").strip()

        if user_input is None:
            print(f"\n⏱️ No input for {input_timeout} seconds. Exiting...")
            break

        if user_input.lower() in {"quit", "exit"}:
            print("\n👋 Session ended.")
            break

        messages.append(HumanMessage(content=user_input))

        # Trim memory
        if len(messages) > max_memory_messages:
            messages = messages[-max_memory_messages:]

        try:
            print("Agent: thinking...", end="", flush=True)

            # Run the graph
            state = agent_app.invoke({"messages": messages})

            # Replace messages with updated state
            messages = state["messages"]

            # Trim again
            if len(messages) > max_memory_messages:
                messages = messages[-max_memory_messages:]

            # Final synthesizer output is ALWAYS the last message
            last = messages[-1]

            print("\rAgent:", last.content, " " * 20, "\n")

        except Exception as e:
            print(f"\nAgent Error: {e}\n")

def run_agent_loop_with_mlflow(agent_app, experiment="/Users/sonalivedaraj@gmail.com/passenger_advocate_agent", max_msgs: int = 20):
    mlflow.set_experiment(experiment)

    print("--- Passenger Advocate Agent ---")
    print("Type 'quit' to exit.\n")

    messages = []
    turn = 0

    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            print("\n👋 Session ended.")
            break

        messages.append(HumanMessage(content=user_input))
        turn += 1

        with mlflow.start_run(run_name=f"turn_{turn}", nested=False):
            start = time.time()

            mlflow.log_param("user_input", user_input)
            mlflow.log_param("messages_before", len(messages))

            # Log conversation history before turn
            mlflow.log_text(
                json.dumps(
                    [{"role": m.__class__.__name__, "content": m.content} for m in messages],
                    indent=2
                ),
                "conversation_before.json"
            )

            print("Agent: thinking...", end="", flush=True)

            if len(messages) > max_msgs:
                messages[-max_msgs:]

            state = agent_app.invoke({"messages": messages})

            messages = state["messages"]
            last = messages[-1]

            latency = time.time() - start
            print("\rAgent:", last.content, " " * 20, "\n")

            # ---- LOG OUTPUT ----
            mlflow.log_param("assistant_output", last.content)
            mlflow.log_metric("latency_seconds", latency)
            mlflow.log_param("messages_after", len(messages))

            # ---- LOG FULL STATE (TOOLS + FLOW TRACE) ----
            try:
                mlflow.log_text(
                    json.dumps(state, default=lambda o: repr(o), indent=2),
                    "agent_state.json"
                )
            except:
                pass

            # ---- TOKEN USAGE (if present) ----
            usage = (
                state.get("usage")
                or state.get("token_usage")
                or getattr(last, "additional_kwargs", {}).get("usage")
            )

            if usage and isinstance(usage, dict):
                for k, v in usage.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f"tokens_{k}", v)

