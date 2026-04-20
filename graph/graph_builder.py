"""
MOSAIC — Multi-Agent Orchestration System for Analyst Intelligence and Cognition
graph/graph_builder.py

Wires all agents and tools into a compiled LangGraph graph.

What the graph is:
    A LangGraph graph is a directed computation graph where each node is
    a Python function that reads from and writes to the shared state.
    Edges define the order of execution — which node runs after which.

    MOSAIC uses a simple sequential topology for reliability:
    All five specialist agents run in sequence, then the Supervisor runs
    last to synthesise their outputs into the final brief.

    We chose sequential over parallel for two reasons:
    1. Rate limits — running 5 agents simultaneously against OpenAI's API
       would hit rate limits on a standard tier immediately
    2. Debuggability — sequential execution makes it trivial to see exactly
       which agent produced which signal in LangSmith traces

    Parallel execution is a straightforward future optimisation once the
    system is running reliably in production.

Graph topology:
    START
      ↓
    temporal_drift_agent    ← language shift across quarters
      ↓
    contradiction_agent     ← 10-K vs earnings call diff
      ↓
    credibility_agent       ← promise vs delivery scoring
      ↓
    supply_chain_agent      ← cross-company signal propagation
      ↓
    insider_signal_agent    ← Form 4 vs call sentiment
      ↓
    supervisor              ← synthesise + compose brief
      ↓
    END

Run standalone? No — imported by main.py or run directly via run_mosaic()
"""

from langgraph.graph import StateGraph, START, END

from graph.state import MosaicState
from agents.temporal_drift_agent import temporal_drift_agent
from agents.contradiction_agent import contradiction_agent
from agents.credibility_agent import credibility_agent
from agents.supply_chain_agent import supply_chain_agent
from agents.insider_signal_agent import insider_signal_agent
from agents.supervisor import supervisor
from config.logging_config import get_logger

logger = get_logger(__name__)


# ── Graph construction ─────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Constructs and compiles the MOSAIC LangGraph graph.

    Adds every agent as a node and connects them in sequential order.
    The graph is compiled once at startup and reused for every run —
    compilation is expensive, execution is cheap.

    Returns:
        Compiled LangGraph graph ready for invocation.
    """
    # Initialise the graph with our shared state schema.
    # Every node in this graph reads from and writes to MosaicState.
    graph = StateGraph(MosaicState)

    # ── Add nodes — one per agent ──────────────────────────────────────────
    # Each node name is a string identifier used in edges and in LangSmith traces.
    # The second argument is the Python function that implements the node.
    graph.add_node("temporal_drift",  temporal_drift_agent)
    graph.add_node("contradiction",   contradiction_agent)
    graph.add_node("credibility",     credibility_agent)
    graph.add_node("supply_chain",    supply_chain_agent)
    graph.add_node("insider_signal",  insider_signal_agent)
    graph.add_node("supervisor",      supervisor)

    # ── Add edges — define execution order ─────────────────────────────────
    # START → first agent
    graph.add_edge(START, "temporal_drift")

    # Sequential chain through all specialist agents
    graph.add_edge("temporal_drift", "contradiction")
    graph.add_edge("contradiction",  "credibility")
    graph.add_edge("credibility",    "supply_chain")
    graph.add_edge("supply_chain",   "insider_signal")

    # Last specialist agent → Supervisor
    graph.add_edge("insider_signal", "supervisor")

    # Supervisor → END
    graph.add_edge("supervisor", END)

    # ── Compile ────────────────────────────────────────────────────────────
    # Compilation validates the graph structure — catches missing nodes,
    # broken edges, and state schema mismatches before any LLM calls are made.
    compiled = graph.compile()

    logger.info(
        "MOSAIC graph compiled — 6 nodes, sequential topology: "
        "temporal_drift → contradiction → credibility → "
        "supply_chain → insider_signal → supervisor"
    )

    return compiled


# ── Module-level compiled graph ────────────────────────────────────────────────

# Compiled once at import time — reused for every run.
# Agents import this directly:
#     from graph.graph_builder import mosaic_graph
#     result = mosaic_graph.invoke(state)
mosaic_graph = build_graph()


# ── Convenience run function ───────────────────────────────────────────────────

def run_mosaic(
    query: str,
    tickers: list[str],
    primary_ticker: str | None = None,
    triggered_by: str = "analyst_request",
) -> MosaicState:
    """
    Runs the full MOSAIC pipeline for a given query and ticker set.

    This is the single entry point for running MOSAIC — it creates
    the initial state, invokes the compiled graph, and returns the
    final state containing all signals, the brief, and the run summary.

    Args:
        query:          The analyst's question or analysis request.
        tickers:        List of companies to analyse.
        primary_ticker: Optional primary focus company.
        triggered_by:   What initiated this run.

    Returns:
        Final MosaicState with all signals, brief, and summary populated.

    Example:
        from graph.graph_builder import run_mosaic

        result = run_mosaic(
            query          = "Analyse MSFT and NVDA for risk signals",
            tickers        = ["MSFT", "NVDA"],
            primary_ticker = "MSFT",
        )

        print(result["final_brief"])
        print(f"Total signals: {len(result['signals'])}")
    """
    from graph.state import create_initial_state
    from processing.vector_store import vector_store

    # Ensure vector store is ready before the graph runs —
    # every agent depends on it and this is cheaper than each
    # agent loading it independently.
    if not vector_store.is_ready:
        logger.info("Loading vector store before graph run")
        vector_store.load()

    # Create clean initial state for this run
    initial_state = create_initial_state(
        query          = query,
        tickers        = tickers,
        triggered_by   = triggered_by,
        primary_ticker = primary_ticker,
    )

    run_id = initial_state["run_id"]

    logger.info(
        f"MOSAIC run starting — "
        f"run_id={run_id} "
        f"tickers={tickers} "
        f"query='{query[:60]}'"
    )

    # Invoke the compiled graph — this runs all 6 nodes sequentially
    # and returns the final merged state after all agents have written to it.
    final_state = mosaic_graph.invoke(initial_state)

    logger.info(
        f"MOSAIC run complete — "
        f"run_id={run_id} "
        f"signals={len(final_state.get('signals', []))} "
        f"hitl={len(final_state.get('hitl_queue', []))} "
        f"risk={final_state.get('run_summary', {}).get('overall_risk', 'unknown')}"
    )

    return final_state