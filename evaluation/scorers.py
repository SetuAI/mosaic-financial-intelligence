"""
MOSAIC — Multi-Agent Orchestration System for Analyst Intelligence and Cognition
evaluation/scorers.py

Custom scoring functions for evaluating MOSAIC signal quality.

What evaluation means in MOSAIC:
    LangSmith provides the infrastructure — it logs every agent run, stores
    inputs and outputs, and lets us run structured evaluations. But LangSmith
    does not know what a good MOSAIC signal looks like. That is what these
    custom scorers define.

    A scorer is a function that takes an agent's output and returns a score
    between 0 and 1 with a reason. LangSmith calls these scorers automatically
    during eval runs and stores the results for comparison.

    We evaluate four dimensions:
    1. Signal completeness  — does the signal have all required fields?
    2. Evidence quality     — is the signal grounded in actual document quotes?
    3. Confidence calibration — is the confidence score appropriate for the finding?
    4. HITL appropriateness — was the HITL routing decision correct?

Why this matters:
    Without evals, we have no systematic way to know if a prompt change made
    the agents better or worse. With evals, every change is measurable.
    "The contradiction agent now correctly identifies metric mismatches
    and has a 23% lower false positive rate" is a statement we can make
    with confidence because we measured it.

Run standalone? No — called by eval_runner.py
"""

from config.logging_config import get_logger

logger = get_logger(__name__)


# ── Signal completeness scorer ─────────────────────────────────────────────────

def score_signal_completeness(output: dict) -> dict:
    """
    Scores whether a signal contains all required fields with non-empty values.

    A complete signal is one where every field that downstream consumers
    depend on is populated. Missing fields cause silent failures in the
    Supervisor and HITL gate.

    Required fields and their importance:
        signal_id        — without this, deduplication fails
        signal_type      — Supervisor groups signals by type
        ticker           — Supervisor groups signals by company
        headline         — shown to analyst in HITL review
        detail           — required for brief composition
        confidence_score — drives HITL routing logic
        severity         — drives urgency classification
        evidence         — required for analyst trust

    Args:
        output: The result dict from an agent node — should contain "signals".

    Returns:
        Dict with score (0-1) and reason string.
    """
    signals = output.get("signals", [])

    if not signals:
        # No signals is valid — the agent found nothing worth flagging.
        # We score this as 1.0 because "no signal" is a complete output.
        return {
            "score":  1.0,
            "reason": "No signals generated — valid output, nothing to score"
        }

    required_fields = [
        "signal_id",
        "signal_type",
        "ticker",
        "headline",
        "detail",
        "confidence_score",
        "severity",
        "evidence",
    ]

    total_checks = 0
    passed       = 0

    for signal in signals:
        for field in required_fields:
            total_checks += 1
            value = signal.get(field)

            # A field passes if it exists and is not empty/None/zero-length
            if value is not None and value != "" and value != []:
                passed += 1
            else:
                logger.debug(
                    f"Signal {signal.get('signal_id', '?')} "
                    f"missing required field: {field}"
                )

    score  = passed / total_checks if total_checks > 0 else 0.0
    reason = (
        f"{passed}/{total_checks} required fields populated "
        f"across {len(signals)} signal(s)"
    )

    return {"score": round(score, 3), "reason": reason}


# ── Evidence quality scorer ────────────────────────────────────────────────────

def score_evidence_quality(output: dict) -> dict:
    """
    Scores whether signals are grounded in actual document evidence.

    A signal without evidence is an opinion. A signal with evidence is
    a finding. The distinction matters enormously for analyst trust —
    "we detected cautious language" is useless without showing which
    specific passages demonstrate the caution.

    Scoring criteria:
    - evidence list is not empty                    → 0.4 points
    - at least one evidence item has a text excerpt → 0.3 points
    - evidence items reference specific filing dates → 0.2 points
    - more than one evidence item provided           → 0.1 points

    Args:
        output: Agent output dict containing "signals".

    Returns:
        Dict with score (0-1) and reason string.
    """
    signals = output.get("signals", [])

    if not signals:
        return {"score": 1.0, "reason": "No signals to evaluate"}

    scores  = []
    reasons = []

    for signal in signals:
        evidence   = signal.get("evidence", [])
        sig_score  = 0.0
        sig_reason = []

        if evidence:
            sig_score += 0.4
            sig_reason.append("evidence list present")
        else:
            sig_reason.append("NO evidence list")

        # Check for text excerpts — actual quotes from source documents
        has_excerpt = any(
            isinstance(e, dict) and (
                e.get("text_excerpt") or
                e.get("call_evidence") or
                e.get("source_excerpt")
            )
            for e in evidence
        )
        if has_excerpt:
            sig_score += 0.3
            sig_reason.append("contains text excerpts")
        else:
            sig_reason.append("no text excerpts found")

        # Check for filing date references — grounds evidence in specific documents
        has_dates = any(
            isinstance(e, dict) and e.get("filing_date")
            for e in evidence
        )
        if has_dates:
            sig_score += 0.2
            sig_reason.append("references specific filing dates")

        # Multiple evidence items is better than one
        if len(evidence) > 1:
            sig_score += 0.1
            sig_reason.append(f"{len(evidence)} evidence items")

        scores.append(sig_score)
        reasons.append(
            f"Signal {signal.get('signal_id', '?')}: "
            f"{', '.join(sig_reason)} → {sig_score:.1f}"
        )

    avg_score = sum(scores) / len(scores) if scores else 0.0

    return {
        "score":  round(avg_score, 3),
        "reason": " | ".join(reasons),
    }


# ── Confidence calibration scorer ─────────────────────────────────────────────

def score_confidence_calibration(output: dict) -> dict:
    """
    Scores whether confidence scores are appropriate for the signal content.

    Over-confident signals (high confidence but weak evidence) erode trust.
    Under-confident signals (low confidence on clear findings) generate
    unnecessary HITL reviews that waste analyst time.

    Calibration checks:
    - Confidence > 0.9 should have multiple evidence items
    - Confidence < 0.6 should always trigger HITL (requires_hitl=True)
    - Critical severity signals should not have confidence > 0.95
      (nothing in financial analysis warrants near-certainty)
    - Insider signals should always have confidence capped ≤ 0.70

    Args:
        output: Agent output dict containing "signals".

    Returns:
        Dict with score (0-1) and reason string.
    """
    signals = output.get("signals", [])

    if not signals:
        return {"score": 1.0, "reason": "No signals to evaluate"}

    issues  = []
    checks  = 0
    passed  = 0

    for signal in signals:
        confidence   = float(signal.get("confidence_score", 0))
        severity     = signal.get("severity", "low")
        requires_hitl = signal.get("requires_hitl", False)
        evidence     = signal.get("evidence", [])
        signal_type  = signal.get("signal_type", "")

        # Check 1: High confidence needs strong evidence
        checks += 1
        if confidence > 0.9 and len(evidence) < 2:
            issues.append(
                f"Signal {signal.get('signal_id', '?')}: "
                f"confidence {confidence} is high but only {len(evidence)} evidence item(s)"
            )
        else:
            passed += 1

        # Check 2: Low confidence should trigger HITL
        checks += 1
        if confidence < 0.6 and not requires_hitl:
            issues.append(
                f"Signal {signal.get('signal_id', '?')}: "
                f"confidence {confidence} is low but HITL not triggered"
            )
        else:
            passed += 1

        # Check 3: Critical severity should not be near-certain
        checks += 1
        if severity == "critical" and confidence > 0.95:
            issues.append(
                f"Signal {signal.get('signal_id', '?')}: "
                f"critical severity with confidence {confidence} — "
                f"nothing in financial analysis warrants near-certainty"
            )
        else:
            passed += 1

        # Check 4: Insider signals must be capped at 0.70
        checks += 1
        if signal_type == "insider_alignment" and confidence > 0.70:
            issues.append(
                f"Insider signal confidence {confidence} exceeds 0.70 cap"
            )
        else:
            passed += 1

    score  = passed / checks if checks > 0 else 0.0
    reason = (
        f"{passed}/{checks} calibration checks passed"
        + (f". Issues: {'; '.join(issues)}" if issues else "")
    )

    return {"score": round(score, 3), "reason": reason}


# ── HITL appropriateness scorer ───────────────────────────────────────────────

def score_hitl_appropriateness(output: dict) -> dict:
    """
    Scores whether HITL routing decisions are appropriate.

    Over-routing to HITL wastes analyst time on signals that do not need
    review. Under-routing publishes signals that should have been reviewed.

    Rules for appropriate HITL routing:
    - Critical severity → must have requires_hitl=True
    - Insider alignment → must have requires_hitl=True (non-negotiable)
    - Confidence < 0.75 → must have requires_hitl=True
    - Confidence >= 0.75 AND severity not critical → should NOT require HITL

    Args:
        output: Agent output dict containing "signals".

    Returns:
        Dict with score (0-1) and reason string.
    """
    signals = output.get("signals", [])

    if not signals:
        return {"score": 1.0, "reason": "No signals to evaluate"}

    checks  = 0
    passed  = 0
    issues  = []

    for signal in signals:
        confidence   = float(signal.get("confidence_score", 0))
        severity     = signal.get("severity", "low")
        requires_hitl = signal.get("requires_hitl", False)
        signal_type  = signal.get("signal_type", "")

        # Critical severity must always go to HITL
        checks += 1
        if severity == "critical" and not requires_hitl:
            issues.append(
                f"Critical signal {signal.get('signal_id', '?')} "
                f"not routed to HITL"
            )
        else:
            passed += 1

        # Insider signals must always go to HITL
        checks += 1
        if signal_type == "insider_alignment" and not requires_hitl:
            issues.append(
                f"Insider signal {signal.get('signal_id', '?')} "
                f"not routed to HITL — mandatory for this type"
            )
        else:
            passed += 1

        # Low confidence must trigger HITL
        checks += 1
        if confidence < 0.75 and not requires_hitl:
            issues.append(
                f"Low confidence signal {signal.get('signal_id', '?')} "
                f"(confidence={confidence}) not routed to HITL"
            )
        else:
            passed += 1

    score  = passed / checks if checks > 0 else 0.0
    reason = (
        f"{passed}/{checks} HITL routing checks passed"
        + (f". Issues: {'; '.join(issues)}" if issues else "")
    )

    return {"score": round(score, 3), "reason": reason}


# ── Composite scorer ───────────────────────────────────────────────────────────

def score_overall_quality(output: dict) -> dict:
    """
    Composite score combining all four dimensions.

    Weights:
        Signal completeness    : 25%
        Evidence quality       : 35%  ← highest weight — evidence is most important
        Confidence calibration : 20%
        HITL appropriateness   : 20%

    Args:
        output: Agent output dict.

    Returns:
        Dict with weighted composite score and breakdown.
    """
    completeness  = score_signal_completeness(output)
    evidence      = score_evidence_quality(output)
    calibration   = score_confidence_calibration(output)
    hitl          = score_hitl_appropriateness(output)

    composite = (
        completeness["score"] * 0.25 +
        evidence["score"]     * 0.35 +
        calibration["score"]  * 0.20 +
        hitl["score"]         * 0.20
    )

    return {
        "score":  round(composite, 3),
        "reason": (
            f"Composite score {composite:.3f} — "
            f"completeness={completeness['score']} "
            f"evidence={evidence['score']} "
            f"calibration={calibration['score']} "
            f"hitl={hitl['score']}"
        ),
        "breakdown": {
            "completeness":  completeness,
            "evidence":      evidence,
            "calibration":   calibration,
            "hitl_routing":  hitl,
        }
    }