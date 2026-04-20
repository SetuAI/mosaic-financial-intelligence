"""
MOSAIC — Multi-Agent Orchestration System for Analyst Intelligence and Cognition
evaluation/eval_runner.py

Runs structured evaluations against MOSAIC agent outputs using LangSmith.

What this file does:
    1. Defines a dataset of test cases — specific queries with expected outputs
    2. Runs each agent against its test cases
    3. Applies custom scorers to measure output quality
    4. Logs everything to LangSmith for comparison and tracking

Why structured evals matter:
    Right now MOSAIC produces good output. But as we iterate — improving
    prompts, adding new tickers, tuning HITL thresholds — we need to know
    whether changes made things better or worse. Without evals, every change
    is a guess. With evals, every change is a measurement.

    A concrete example: if we tighten the contradiction agent's system prompt
    to reduce false positives, the eval runner will tell us whether the
    true positive rate held steady or dropped. That is the difference between
    "I think this is better" and "this is measurably better".

LangSmith integration:
    LangSmith's client reads LANGCHAIN_API_KEY directly from os.environ —
    not from pydantic settings. We set it explicitly before any LangSmith
    calls are made to ensure the key is available to the library.

    Each eval run creates a named experiment in LangSmith that you can
    view at smith.langchain.com. Runs are comparable — you can see
    side by side how score distributions changed between prompt versions.

Run standalone? Yes
Command: python -m evaluation.eval_runner
         python -m evaluation.eval_runner --no-langsmith
         python -m evaluation.eval_runner --case msft_contradiction
"""

import os
import argparse

from datetime import datetime
from typing import Optional

from config.settings import settings
from config.logging_config import setup_logging, get_logger
from evaluation.scorers import (
    score_signal_completeness,
    score_evidence_quality,
    score_confidence_calibration,
    score_hitl_appropriateness,
    score_overall_quality,
)

logger = get_logger(__name__)


# ── Test cases ─────────────────────────────────────────────────────────────────

# Each test case defines a query, the tickers to analyse, and the
# minimum acceptable scores for each scoring dimension.
# These are the baseline expectations — a score below minimum is a regression.
EVAL_TEST_CASES = [
    {
        "name":           "msft_temporal_drift",
        "description":    "Temporal drift analysis should detect language shift in MSFT filings",
        "agent":          "temporal_drift",
        "query":          "Analyse MSFT language drift across last 2 years of filings",
        "tickers":        ["MSFT"],
        "primary_ticker": "MSFT",
        "min_scores": {
            "completeness":  0.90,
            "evidence":      0.50,
            "calibration":   0.75,
            "hitl_routing":  1.00,
        },
    },
    {
        "name":           "msft_contradiction",
        "description":    "Contradiction agent should find gaps between MSFT call and 10-K",
        "agent":          "contradiction",
        "query":          "Find contradictions between MSFT earnings calls and 10-K filings",
        "tickers":        ["MSFT"],
        "primary_ticker": "MSFT",
        "min_scores": {
            "completeness":  0.90,
            "evidence":      0.60,
            "calibration":   0.75,
            "hitl_routing":  1.00,
        },
    },
    {
        "name":           "nvda_supply_chain",
        "description":    "Supply chain agent should detect NVDA stress and propagation",
        "agent":          "supply_chain",
        "query":          "Identify supply chain stress signals for NVDA",
        "tickers":        ["NVDA", "AAPL", "MSFT"],
        "primary_ticker": "NVDA",
        "min_scores": {
            "completeness":  0.90,
            "evidence":      0.50,
            "calibration":   0.75,
            "hitl_routing":  1.00,
        },
    },
    {
        "name":           "msft_credibility",
        "description":    "Credibility agent should score MSFT management accurately",
        "agent":          "credibility",
        "query":          "Score MSFT management credibility on guidance delivery",
        "tickers":        ["MSFT"],
        "primary_ticker": "MSFT",
        "min_scores": {
            "completeness":  0.85,
            "evidence":      0.40,
            "calibration":   0.75,
            "hitl_routing":  1.00,
        },
    },
    {
        "name":           "insider_alignment",
        "description":    "Insider agent should detect cluster selling and always HITL",
        "agent":          "insider_signal",
        "query":          "Analyse insider transactions vs earnings call sentiment for MSFT",
        "tickers":        ["MSFT"],
        "primary_ticker": "MSFT",
        "min_scores": {
            "completeness":  0.85,
            "evidence":      0.40,
            # Insider signals must always be capped at 0.70 confidence
            "calibration":   0.75,
            # Insider signals must always route to HITL — non-negotiable
            "hitl_routing":  1.00,
        },
    },
]


# ── Agent runner ───────────────────────────────────────────────────────────────

def run_agent_for_eval(
    agent_name: str,
    query: str,
    tickers: list[str],
    primary_ticker: Optional[str] = None,
) -> dict:
    """
    Runs a specific agent and returns its output for evaluation.

    Imports agents lazily so the eval runner can be imported without
    triggering all agent imports at once — useful for running targeted
    evals on a single agent without loading the full system.

    Args:
        agent_name:     Name of the agent to run.
        query:          The analysis query for this eval case.
        tickers:        Tickers to analyse.
        primary_ticker: Primary focus ticker.

    Returns:
        Agent output dict — the partial state update the agent returned.
    """
    from graph.state import create_initial_state
    from processing.vector_store import vector_store

    if not vector_store.is_ready:
        vector_store.load()

    state = create_initial_state(
        query          = query,
        tickers        = tickers,
        primary_ticker = primary_ticker,
    )

    agent_map = {
        "temporal_drift": _run_temporal_drift,
        "contradiction":  _run_contradiction,
        "credibility":    _run_credibility,
        "supply_chain":   _run_supply_chain,
        "insider_signal": _run_insider_signal,
    }

    if agent_name not in agent_map:
        raise ValueError(
            f"Unknown agent '{agent_name}'. "
            f"Valid options: {list(agent_map.keys())}"
        )

    return agent_map[agent_name](state)


def _run_temporal_drift(state):
    from agents.temporal_drift_agent import temporal_drift_agent
    return temporal_drift_agent(state)

def _run_contradiction(state):
    from agents.contradiction_agent import contradiction_agent
    return contradiction_agent(state)

def _run_credibility(state):
    from agents.credibility_agent import credibility_agent
    return credibility_agent(state)

def _run_supply_chain(state):
    from agents.supply_chain_agent import supply_chain_agent
    return supply_chain_agent(state)

def _run_insider_signal(state):
    from agents.insider_signal_agent import insider_signal_agent
    return insider_signal_agent(state)


# ── Eval runner ────────────────────────────────────────────────────────────────

def run_evals(
    test_cases: Optional[list[dict]] = None,
    use_langsmith: bool = True,
) -> list[dict]:
    """
    Runs all eval test cases and scores the outputs.

    When LangSmith tracing is enabled, results appear as a named experiment
    at smith.langchain.com with full input/output traces and score distributions.

    When use_langsmith is False, runs locally and prints results to terminal.
    Useful for quick iteration during development without API credentials.

    Args:
        test_cases:    Test cases to run. Defaults to EVAL_TEST_CASES.
        use_langsmith: Whether to log to LangSmith. Default True.

    Returns:
        List of eval result dicts with scores for each test case.
    """
    cases    = test_cases or EVAL_TEST_CASES
    results  = []
    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    if use_langsmith:
        # LangSmith's client reads directly from os.environ — not from
        # our pydantic settings object. Setting it here ensures the key
        # is available before any LangSmith calls are made.
        os.environ["LANGCHAIN_API_KEY"]     = settings.langchain_api_key
        os.environ["LANGCHAIN_TRACING_V2"]  = "true"
        os.environ["LANGCHAIN_PROJECT"]     = f"mosaic-eval-{run_time}"

        logger.info(
            f"LangSmith tracing enabled — "
            f"project: mosaic-eval-{run_time}"
        )
    else:
        # Explicitly disable tracing — prevents stale env vars from
        # a previous session accidentally enabling it
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        logger.info("Running evals locally — LangSmith tracing disabled")

    logger.info(f"Running {len(cases)} eval test cases")

    for case in cases:
        logger.info(f"\nRunning eval: {case['name']}")
        logger.info(f"  Agent   : {case['agent']}")
        logger.info(f"  Tickers : {case['tickers']}")

        try:
            # Run the agent
            output = run_agent_for_eval(
                agent_name     = case["agent"],
                query          = case["query"],
                tickers        = case["tickers"],
                primary_ticker = case.get("primary_ticker"),
            )

            # Score the output
            scores = _score_output(output)

            # Check against minimum thresholds
            min_scores  = case.get("min_scores", {})
            regressions = _check_regressions(scores, min_scores)

            result = {
                "test_case":   case["name"],
                "agent":       case["agent"],
                "tickers":     case["tickers"],
                "signals":     len(output.get("signals", [])),
                "scores":      scores,
                "regressions": regressions,
                "passed":      len(regressions) == 0,
            }

            results.append(result)

            status = "✓ PASS" if result["passed"] else "✗ FAIL"
            logger.info(f"  {status} — {case['name']}")
            logger.info(f"  Signals: {result['signals']}")
            logger.info(f"  Scores: {_format_scores(scores)}")

            if regressions:
                logger.warning(f"  Regressions: {regressions}")

        except Exception as e:
            logger.error(f"  ERROR running {case['name']}: {e}", exc_info=True)
            results.append({
                "test_case":   case["name"],
                "agent":       case["agent"],
                "error":       str(e),
                "passed":      False,
                "regressions": ["EXECUTION_ERROR"],
            })

    _print_eval_summary(results)
    return results


# ── Scoring helpers ────────────────────────────────────────────────────────────

def _score_output(output: dict) -> dict:
    """
    Applies all scorers to an agent output and returns the score dict.

    Args:
        output: Agent output dict from run_agent_for_eval().

    Returns:
        Dict of scorer_name -> score_result.
    """
    return {
        "completeness": score_signal_completeness(output),
        "evidence":     score_evidence_quality(output),
        "calibration":  score_confidence_calibration(output),
        "hitl_routing": score_hitl_appropriateness(output),
        "overall":      score_overall_quality(output),
    }


def _check_regressions(scores: dict, min_scores: dict) -> list[str]:
    """
    Identifies scores that fell below their minimum thresholds.

    Args:
        scores:     Actual scores from _score_output().
        min_scores: Minimum acceptable scores from the test case definition.

    Returns:
        List of regression descriptions, empty if all thresholds met.
    """
    regressions = []

    score_map = {
        "completeness": scores.get("completeness", {}).get("score", 0),
        "evidence":     scores.get("evidence", {}).get("score", 0),
        "calibration":  scores.get("calibration", {}).get("score", 0),
        "hitl_routing": scores.get("hitl_routing", {}).get("score", 0),
    }

    for metric, minimum in min_scores.items():
        actual = score_map.get(metric, 0)
        if actual < minimum:
            regressions.append(
                f"{metric}: {actual:.3f} < {minimum:.3f} (minimum)"
            )

    return regressions


def _format_scores(scores: dict) -> str:
    """Formats scores as a compact string for logging."""
    parts = []
    for key in ["completeness", "evidence", "calibration", "hitl_routing"]:
        score = scores.get(key, {}).get("score", 0)
        parts.append(f"{key}={score:.2f}")
    return " | ".join(parts)


def _print_eval_summary(results: list[dict]) -> None:
    """Prints a summary table of all eval results."""
    passed = sum(1 for r in results if r.get("passed", False))
    failed = len(results) - passed

    print("\n" + "=" * 60)
    print("MOSAIC EVAL SUMMARY")
    print("=" * 60)
    print(f"Total : {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print()

    for r in results:
        status  = "✓ PASS" if r.get("passed") else "✗ FAIL"
        scores  = r.get("scores", {})
        overall = scores.get("overall", {}).get("score", 0)
        print(
            f"  {status} | {r['test_case']:30s} | "
            f"signals={r.get('signals', 'ERR'):3} | "
            f"overall={overall:.2f}"
        )
        if r.get("regressions"):
            for reg in r["regressions"]:
                print(f"         REGRESSION: {reg}")

    print("=" * 60)

    if failed == 0:
        print("All evals passed — no regressions detected.")
    else:
        print(
            f"{failed} eval(s) failed. "
            f"Review regressions before deploying prompt changes."
        )


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    setup_logging()

    parser = argparse.ArgumentParser(
        description="MOSAIC evaluation runner",
        epilog="""
Examples:
  python -m evaluation.eval_runner                     # all cases, LangSmith on
  python -m evaluation.eval_runner --no-langsmith      # local only, no tracing
  python -m evaluation.eval_runner --case msft_contradiction
        """,
    )

    parser.add_argument(
        "--no-langsmith",
        action="store_true",
        help="Disable LangSmith tracing — run locally only",
    )

    parser.add_argument(
        "--case",
        help="Run a single named test case",
        metavar="CASE_NAME",
    )

    args = parser.parse_args()

    # Filter to a specific case if requested
    cases = EVAL_TEST_CASES
    if args.case:
        cases = [c for c in EVAL_TEST_CASES if c["name"] == args.case]
        if not cases:
            print(f"Test case '{args.case}' not found.")
            print(f"Available: {[c['name'] for c in EVAL_TEST_CASES]}")
            exit(1)

    run_evals(
        test_cases    = cases,
        use_langsmith = not args.no_langsmith,
    )