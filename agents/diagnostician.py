"""
Diagnostician Agent.

Receives the raw Cantera result + current mechanism context.
Returns a structured DiagnosticianReport identifying:
  - failure_mode: what went wrong chemically (named by chain stage)
  - missing_pathways: specific reactions suspected missing
  - target_species: what to search for next
  - severity: fatal | partial | ok

The LLM reasons through the combustion chain systematically, not by pattern matching.
It must ONLY reference reactions already in the mechanism.
"""

import json
import logging
from agents.state import OrchestratorState, DiagnosticianReport

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an expert combustion chemist diagnosing a failed kinetic mechanism.
You reason systematically through the combustion chain, not by guessing or pattern matching.

## The Combustion Chain — your diagnostic checklist

For a combustion mechanism to ignite, ALL five stages must be present and balanced:

### Stage 1 — Chain Initiation
  What it does: creates first radicals from stable molecules
  Reactions:    unimolecular decomposition (fuel + M → radicals) or
                bimolecular initiation (fuel + O2 → radical + HO2)
  Failure sign: score = 0%, no CVodes error — mechanism is chemically inert
  Family:       Decomposition, H_Abstraction

### Stage 2 — Chain Branching
  What it does: 1 radical → 2 radicals, drives exponential radical buildup → ignition
  Key reaction: H + O2 → OH + O  (for H2/O2 and any H-containing fuel)
                or fuel_radical + O2 → 2 radicals
  Failure sign: score = 0% even with initiation present; no temperature spike
  Family:       H_Abstraction, Other

### Stage 3 — Chain Propagation (H_Abstraction)
  What it does: OH/O radicals abstract H from fuel, regenerating H for branching
  Key reactions: OH + fuel → H2O + fuel_radical
                 O + fuel → OH + fuel_radical
  Failure sign: high-T conditions fail (high-T ignition relies heavily on OH + fuel)
  Family:       H_Abstraction

### Stage 4 — Peroxy / HO2 Chemistry
  What it does: manages HO2 and H2O2 pool; critical at low-T and high-P
  Key reactions: H + O2 + M → HO2 + M    (ThirdBody/Troe — competes with branching)
                 HO2 + H → 2 OH           (re-generates chain carriers from HO2)
                 H2O2 + M → 2 OH + M      (Troe — liberates OH at intermediate T)
                 HO2 + OH → H2O + O2      (Disproportionation)
  Failure sign: low-T conditions fail, high-P conditions fail; HO2 accumulates
  Family:       R_Recombination (H+O2+M), Disproportionation, Decomposition

### Stage 5 — Chain Termination (R_Recombination)
  What it does: consumes radical pairs, prevents radical runaway, stabilises ODE
  Key reactions: H + OH + M → H2O + M     (most important — removes H and OH together)
                 H + H + M → H2 + M
                 H + O2 + M → HO2 + M     (also counts — converts H to stable HO2)
  Failure sign: CVodes error code -3 ("error test failed" / "too much work")
               = radical pool diverges = termination is ABSENT
               This is NOT a "hard" chemistry problem — it is a STRUCTURAL gap.
  Family:       R_Recombination

## DIAGNOSTIC PROCEDURE — follow this order

1. **Check CVodes error**:
   If ANY condition shows "error test failed" or "too much work" or code -3:
   → failure_mode = "Stage 5 MISSING: no chain termination (R_Recombination)"
   → missing_pathways = ["H + OH + M <=> H2O + M", "H + H + M <=> H2 + M"]
   → severity = "fatal"
   → target_species = ["H", "OH", "H2O"]
   STOP — do not analyse further, these reactions are the fix.

2. **Check score = 0% without CVodes error**:
   → Look for Stage 1 and Stage 2 in current_reaction_labels
   → If Stage 2 (H + O2 → OH + O or equivalent) is absent:
     failure_mode = "Stage 2 MISSING: no chain branching — mechanism cannot ignite"
   → If Stage 2 is present but score = 0:
     failure_mode = "Stage 1 MISSING: no chain initiation — radical pool never starts"

3. **Check partial score (some conditions pass, some fail)**:
   → Identify which T/P conditions fail vs pass from idt_results
   → If LOW-T fail, HIGH-T pass: Stage 4 gap (peroxy chemistry missing)
   → If HIGH-T fail, LOW-T pass: Stage 2 or Stage 3 gap
   → If HIGH-P fail, LOW-P pass: Stage 4 ThirdBody reactions missing
   → Use flux analysis sensitivity data to confirm which reactions are rate-limiting

4. **If flux analysis is available** (score > 0):
   Trust the sensitivity data — it is Cantera's own measurement, not intuition.
   The most-negative sensitivity reactions are rate-limiting for ignition.
   If any of those are NOT in current_reaction_labels, that is what to add.
   Use `isolated_species` to identify species with no pathway.
   Use `missing_families` to confirm structural gaps.

## STRICT RULES
- You may ONLY reference reactions that are in the current_reaction_labels list
- You may NOT invent rate constants, Arrhenius parameters, or kinetic values
- You may NOT suggest reactions already in current_reaction_labels
- If you are uncertain, name the most likely gap — do not output empty missing_pathways
- Your missing_pathways must be chemically sensible reaction labels

## Output schema (JSON ONLY — no markdown, no preamble)
{
  "chain_audit": {
    "stage1_initiation": "present / ABSENT — evidence from current_reaction_labels",
    "stage2_branching":  "present / ABSENT — evidence",
    "stage3_propagation": "present / ABSENT — evidence",
    "stage4_peroxy":     "present / ABSENT — evidence",
    "stage5_termination": "present / ABSENT — evidence"
  },
  "failure_mode": "Stage N MISSING: specific description of the gap",
  "missing_pathways": ["reaction label that should be added", ...],
  "target_species": ["species whose chemistry is incomplete"],
  "severity": "fatal" | "partial" | "ok",
  "reasoning": "step-by-step chain analysis (string)"
}
"""


def build_prompt(context: dict) -> str:
    """Build the diagnostician's user prompt from orchestrator context."""
    lines = [
        f"## Iteration {context['iteration']}",
        f"## Fuel: {context.get('fuel', '?')} / Oxidizer: {context.get('oxidizer', 'O2')}",
        f"",
        f"## Current mechanism reactions ({len(context['current_reaction_labels'])} total)",
        f"## (These are annotated: [three-body → rendered with +M in YAML] means ThirdBody/Troe/Lindemann)",
    ]
    for label in context["current_reaction_labels"]:
        lines.append(f"  - {label}")

    lines += [
        f"",
        f"## Current species: {context['current_species']}",
        f"",
        f"## Cantera validation score: {context['validation_score']:.1%}",
        f"",
        f"## IDT results per condition:",
    ]
    for r in context["idt_results"]:
        status = "PASS" if r["success"] and r.get("idt_ms") else "FAIL"
        error  = f" — {r['error'][:160]}" if r.get("error") else ""
        idt    = f" IDT={r['idt_ms']:.3f}ms" if r.get("idt_ms") else ""
        lines.append(f"  {r['condition']}: {status}{idt}{error}")

    lines += ["", "## Validation history:"]
    for v in context["validation_history"]:
        lines.append(f"  iter {v['iteration']}: score={v['score']:.1%}")

    if context.get("last_addition"):
        lines += [
            f"",
            f"## Last addition:",
            f"  Added: {context['last_addition']['added']}",
            f"  Reason: {context['last_addition']['reason']}",
        ]

    # Flux analysis — trust Cantera's own measurements
    flux = context.get("flux_analysis", {})
    if flux.get("top_sensitive_reactions"):
        lines += [
            f"",
            f"## Cantera Flux Analysis (Cantera's own measurements — trust these over intuition)",
            f"### Sensitivity to ignition delay (most negative = most rate-limiting for ignition)",
        ]
        for r in flux["top_sensitive_reactions"][:10]:
            lines.append(f"  {r['sensitivity']:+.4f}  {r['reaction']}")

        if flux.get("isolated_species"):
            lines += [
                f"",
                f"### Structurally isolated species (< 2 reactions — missing pathways)",
                f"  {flux['isolated_species']}",
            ]

        cov = flux.get("family_coverage", {})
        if cov:
            lines += [
                f"",
                f"### Reaction family coverage (RMG-style chain completeness)",
                f"  Covered families:  {cov.get('covered_families', [])}",
                f"  MISSING families:  {cov.get('missing_families', [])}",
                f"  Coverage:          {cov.get('coverage_pct', '?')}% of expected families",
            ]
            for rec in cov.get("recommendations", []):
                lines.append(f"  Recommendation: {rec}")
    else:
        lines += [
            f"",
            f"## Cantera Flux Analysis: not available (score = 0 — mechanism does not ignite)",
            f"  → Use the chain audit procedure to identify what is missing structurally.",
        ]

    lines += [
        f"",
        f"Now: audit the mechanism by chain stage (1→5), identify the failure, output JSON.",
    ]
    return "\n".join(lines)


def parse_and_validate(raw: str, state: OrchestratorState) -> DiagnosticianReport:
    """Parse JSON output and validate against known constraints."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error(f"Diagnostician JSON parse failed: {e}\nRaw: {raw[:300]}")
        # Fallback: assume chain termination is missing (most common fatal issue)
        return DiagnosticianReport(
            failure_mode="Stage 5 MISSING: no chain termination — JSON parse error, using safe fallback",
            missing_pathways=["H + OH + M <=> H2O + M", "H + H + M <=> H2 + M"],
            target_species=["H", "OH", "H2O", "HO2"],
            severity="fatal",
            raw_cantera_error=str(e),
        )

    failure_mode     = data.get("failure_mode", "unknown chain failure")
    missing_pathways = data.get("missing_pathways", [])
    target_species   = data.get("target_species", [])
    severity         = data.get("severity", "partial")

    if severity not in ("fatal", "partial", "ok"):
        severity = "partial"

    # Log chain audit if present
    audit = data.get("chain_audit", {})
    if audit:
        for stage, status in audit.items():
            logger.info(f"  {stage}: {status}")

    logger.info(f"Diagnostician: {failure_mode} | severity={severity}")
    logger.info(f"  Missing: {missing_pathways}")
    logger.info(f"  Targets: {target_species}")

    return DiagnosticianReport(
        failure_mode=failure_mode,
        missing_pathways=missing_pathways,
        target_species=target_species,
        severity=severity,
    )


def run(state: OrchestratorState, llm) -> DiagnosticianReport:
    """
    Run the diagnostician agent.

    Args:
        state: Current orchestrator state (read-only)
        llm:   LLMOrchestrator instance (owned by this agent)

    Returns:
        DiagnosticianReport
    """
    context = state.context_for_diagnostician()
    prompt  = build_prompt(context)

    logger.info(f"[Diagnostician] Running diagnosis — iter {state.iteration}")

    raw = llm.chat(
        messages=[{"role": "user", "content": prompt}],
        system_prompt=SYSTEM_PROMPT,
    )

    logger.debug(f"[Diagnostician] Raw output: {raw[:600]}")
    report = parse_and_validate(raw, state)
    return report
