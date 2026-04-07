"""
Chemistry Selector Agent.

The RMG-equivalent brain. Given:
  - The full candidate pool (from DB search, already classified by family)
  - The current mechanism state and family coverage
  - The diagnostician's report

It selects 1-3 reactions to add THIS iteration using systematic chain analysis.

Chain analysis approach (mirrors RMG's core/edge promotion logic):
  1. Identify which chain stages are covered (initiation, branching, propagation, termination)
  2. Find the most critical gap (what makes CVodes fail or score stay low?)
  3. Select reactions that fill that gap from the correct family

CRITICAL constraint: selections are by INDEX into the candidate pool (0-based).
This prevents label-hallucination. The parser resolves the actual label from the index.
"""

import json
import logging
from collections import defaultdict
from agents.state import OrchestratorState, SelectorOutput

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Chain role descriptions — maps reaction family → combustion chain role
# ─────────────────────────────────────────────────────────────────────────────

FAMILY_CHAIN_ROLES = {
    "R_Recombination":          "CHAIN TERMINATION — radical + radical (+M) → stable; PREVENTS CVodes divergence",
    "H_Abstraction":            "CHAIN PROPAGATION — H-atom transfer; sustains the radical pool",
    "Decomposition":            "INTERMEDIATE BREAKDOWN — H2O2/HO2 → radicals; critical at low-T",
    "Disproportionation":       "RADICAL QUENCHING — radical + radical → 2 stable products",
    "R_Addition_MultipleBond":  "RADICAL ADDITION — radical + unsaturated → adduct (peroxy at high P)",
    "Intra_H_migration":        "ISOMERIZATION — internal H-shift (important for large fuels)",
    "Other":                    "OTHER — uncategorized",
}

# Priority order for displaying candidates (most critical families first)
FAMILY_DISPLAY_ORDER = [
    "R_Recombination",
    "H_Abstraction",
    "Decomposition",
    "Disproportionation",
    "R_Addition_MultipleBond",
    "Intra_H_migration",
    "Other",
]

# ─────────────────────────────────────────────────────────────────────────────
# System prompt — teaches chain analysis reasoning
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert combustion chemist building a kinetic mechanism from scratch.
You think systematically through the combustion chain, exactly as RMG does when growing its core mechanism.

## The Combustion Chain — you MUST reason through ALL stages before selecting

Every combustion mechanism needs these chain stages, in order of priority:

### Stage 1 — Chain Initiation (kinetics: Arrhenius or ThirdBody)
  Purpose: generate first radicals from stable fuel + oxidizer
  Required: at least ONE reaction that produces H, O, or OH from stable reactants
  Family:   Decomposition (unimolecular), H_Abstraction (bimolecular)
  Example:  H2 + O2 → HO2 + H   |   H2O2 + M → OH + OH + M

### Stage 2 — Chain Branching (kinetics: Arrhenius)
  Purpose: 1 radical → 2 radicals, exponentially grows radical pool → ignition
  Required: H + O2 → OH + O  (THE single most important H2/O2 reaction)
            or equivalent fuel-radical + O2 → 2 radicals
  Family:   H_Abstraction or Other
  WARNING:  Without this, ignition CANNOT occur. Score stays at 0%.

### Stage 3 — Chain Propagation (kinetics: Arrhenius)
  Purpose: radicals consume fuel and regenerate chain carriers
  Required: OH + fuel → H2O + fuel_radical   |   O + fuel → OH + fuel_radical
  Family:   H_Abstraction
  Example:  OH + H2 → H + H2O   |   O + H2 → OH + H

### Stage 4 — Peroxy/HO2 Chemistry (kinetics: Arrhenius or ThirdBody)
  Purpose: HO2 formation and consumption at intermediate T and high P
  Required: H + O2 + M → HO2 + M  (ThirdBody/Troe — competes with branching at high P)
            HO2 + H → OH + OH   |   HO2 + OH → H2O + O2
  Family:   R_Recombination (H+O2+M), H_Abstraction or Disproportionation (HO2 reactions)

### Stage 5 — Chain Termination (kinetics: ThirdBody or Troe) — HIGHEST PRIORITY
  Purpose: radical + radical → stable product; prevents radical runaway
  Required: H + OH + M → H2O + M   |   H + H + M → H2 + M   |   H + O + M → OH + M
  Family:   R_Recombination
  CRITICAL: CVodes error code -3 (error test failed) = radical runaway = Stage 5 is MISSING.
            If you see CVodes -3, your FIRST selection MUST be an R_Recombination reaction.

## How to analyze and select

Step 1 — AUDIT the current mechanism by stage:
  For each stage above, check current_reaction_labels. Is that stage covered?
  A stage is "covered" if there is at least one reaction in the mechanism serving that role.

Step 2 — IDENTIFY the critical gap:
  - CVodes error -3 → Stage 5 (R_Recombination) is missing — select from that family FIRST
  - Score = 0, no CVodes error → Stage 1 or Stage 2 is missing
  - Score partial, low-T fails → Stage 4 (HO2/peroxy) is incomplete
  - Score partial, high-T fails → Stage 2 or Stage 3 gap
  - Score partial, high-P fails → Stage 4 ThirdBody reactions missing

Step 3 — SELECT from the correct family:
  Look at the candidate pool section matching the critical gap's family.
  Pick the reaction with the best match to what is missing.
  Use the [NNN] integer index in your JSON output — NOT the label text.

Step 4 — VERIFY your selection:
  - Is the index valid (0 to pool_size-1)?
  - Is the reaction NOT already in current_reaction_labels?
  - Does it fill a real gap, not a gap that is already covered?

## ABSOLUTE RULES
1. Use the INTEGER INDEX [NNN] from the candidate pool — do NOT copy label text
2. Do NOT re-select reactions already in current_reaction_labels
3. Your reasoning MUST name which chain stage each selection addresses
4. Output valid JSON matching the schema exactly

## Output schema (JSON ONLY — no markdown, no preamble)
{
  "chain_analysis": {
    "stage1_initiation": "covered / MISSING — reasoning",
    "stage2_branching":  "covered / MISSING — reasoning",
    "stage3_propagation": "covered / MISSING — reasoning",
    "stage4_peroxy":     "covered / MISSING — reasoning",
    "stage5_termination": "covered / MISSING — reasoning",
    "critical_gap": "which stage is most urgently needed"
  },
  "selected": [
    {
      "index": 42,
      "chain_stage": "Stage N — role description",
      "reason": "specific chemical reason: WHY this reaction fills the gap given current mechanism state"
    }
  ],
  "reasoning": "overall chain-of-thought",
  "next_target_species": ["species to prioritise in next DB search"]
}
"""


def build_prompt(context: dict, candidate_pool: list[dict]) -> str:
    """Build full selector prompt with candidates organized by reaction family."""
    lines = [
        f"## Iteration {context['iteration']}",
        f"## Fuel: {context['fuel']} / Oxidizer: {context['oxidizer']}",
        f"## Select {context['n_to_select']} reaction(s) to add next",
        f"",
        f"## Species with available thermodynamic data",
        f"  (reactions with species NOT in this set will be skipped at build time)",
        f"  {context['known_thermo_species']}",
        f"",
        f"## Diagnostician report",
        f"  Failure mode: {context['diagnostician_report']['failure_mode']}",
        f"  Missing pathways: {context['diagnostician_report']['missing_pathways']}",
        f"  Target species:   {context['diagnostician_report']['target_species']}",
        f"  Severity:         {context['diagnostician_report']['severity']}",
        f"",
        f"## Current mechanism ({context['n_current_reactions']} reactions)",
        f"## (DO NOT re-select any of these)",
    ]
    for label in context["current_reaction_labels"]:
        lines.append(f"  ✓ {label}")

    lines += ["", "## Addition log (what was added and why in previous iterations)"]
    if context["addition_log"]:
        for entry in context["addition_log"]:
            lines.append(f"  iter {entry['iteration']}: {entry['added']} — {entry['reason']}")
    else:
        lines.append("  (first iteration — nothing added yet)")

    lines += ["", "## Validation history"]
    for v in context["validation_history"]:
        lines.append(f"  iter {v['iteration']}: score={v['score']:.1%} ({v['failure_mode']})")

    # Family coverage summary (from flux analysis if available)
    family_cov = context.get("family_coverage", {})
    if family_cov:
        lines += ["", "## Reaction family coverage (RMG-style)"]
        lines.append(f"  Covered:  {family_cov.get('covered_families', [])}")
        lines.append(f"  MISSING:  {family_cov.get('missing_families', [])}")
        lines.append(f"  Coverage: {family_cov.get('coverage_pct', '?')}% of expected families")
        for rec in family_cov.get("recommendations", []):
            lines.append(f"  → {rec}")

    # ── Candidate pool organized by family ───────────────────────────────────
    lines += [
        "",
        f"## Candidate pool — {context['candidate_pool_count']} reactions organized by CHAIN ROLE",
        "## Use the integer [NNN] INDEX in your JSON 'index' field (NOT the label text)",
        "",
    ]

    # Group candidates by family
    by_family: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for global_idx, rxn in enumerate(candidate_pool):
        fam = rxn.get("family", "Other")
        by_family[fam].append((global_idx, rxn))

    for family in FAMILY_DISPLAY_ORDER:
        entries = by_family.get(family, [])
        if not entries:
            continue
        role = FAMILY_CHAIN_ROLES.get(family, "")
        lines.append(f"### {family} — {role} ({len(entries)} candidates)")
        for global_idx, rxn in entries:
            kt = rxn.get("kinetics_type", "?")
            lib = rxn.get("library", "?")
            lines.append(f"  [{global_idx:03d}] {rxn['label']}  ({kt})  lib={lib}")
        lines.append("")

    # Any families not in FAMILY_DISPLAY_ORDER
    for family, entries in by_family.items():
        if family not in FAMILY_DISPLAY_ORDER:
            lines.append(f"### {family} ({len(entries)} candidates)")
            for global_idx, rxn in entries:
                lines.append(f"  [{global_idx:03d}] {rxn['label']}  ({rxn.get('kinetics_type','?')})  lib={rxn.get('library','?')}")
            lines.append("")

    lines += [
        f"",
        f"Now: audit the current mechanism by chain stage, identify the critical gap,",
        f"and select {context['n_to_select']} reaction(s) by their [NNN] index.",
        f"Output JSON only.",
    ]
    return "\n".join(lines)


def parse_and_validate(
    raw: str,
    candidate_pool: list[dict],
    current_labels: set,
    n_to_select: int,
) -> SelectorOutput:
    """
    Parse JSON output and resolve index → reaction.
    Supports both index-based (primary) and label-based (legacy) selection.
    """
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error(f"Selector JSON parse failed: {e}")
        fallback = _fallback_selection(candidate_pool, current_labels, n_to_select)
        return SelectorOutput(
            selected=fallback,
            reasoning=f"JSON parse error — fallback selection used: {e}",
            next_target_species=[],
        )

    selected_raw = data.get("selected", [])
    validated = []
    rejected  = []

    for item in selected_raw:
        reason = item.get("reason", "no reason given")

        # Primary path: index-based selection
        if "index" in item:
            idx = item["index"]
            if not isinstance(idx, int) or idx < 0 or idx >= len(candidate_pool):
                rejected.append(f"INVALID INDEX: {idx!r}")
                logger.warning(f"Selector gave invalid index: {idx!r}")
                continue
            rxn = candidate_pool[idx]
            label = rxn["label"]
        else:
            # Legacy path: label string (kept for compatibility)
            label = item.get("label", "")
            pool_labels = {r["label"] for r in candidate_pool}
            if label not in pool_labels:
                rejected.append(f"HALLUCINATED (not in pool): {label!r}")
                logger.warning(f"Selector hallucinated label: {label!r}")
                continue
            rxn = next(r for r in candidate_pool if r["label"] == label)

        if label in current_labels:
            rejected.append(f"ALREADY IN MECHANISM: {label!r}")
            logger.warning(f"Selector tried to re-add: {label!r}")
            continue

        full_rxn = dict(rxn)
        full_rxn["_selection_reason"] = reason
        full_rxn["_chain_stage"] = item.get("chain_stage", "")
        validated.append(full_rxn)

    if rejected:
        logger.warning(f"Selector rejections: {rejected}")

    if not validated:
        logger.warning("All selector choices were invalid — using fallback")
        validated = _fallback_selection(candidate_pool, current_labels, n_to_select)

    validated = validated[:n_to_select]

    # Log chain analysis if present
    chain = data.get("chain_analysis", {})
    if chain:
        gap = chain.get("critical_gap", "unknown")
        logger.info(f"[Selector] Chain analysis — critical gap: {gap}")

    logger.info(f"[Selector] Selected {len(validated)} reactions: {[r['label'] for r in validated]}")

    return SelectorOutput(
        selected=validated,
        reasoning=data.get("reasoning", ""),
        next_target_species=data.get("next_target_species", []),
    )


def _fallback_selection(
    candidate_pool: list[dict],
    current_labels: set,
    n: int,
) -> list[dict]:
    """
    Fallback when LLM output is invalid.
    Priority: R_Recombination (chain termination) first, then H_Abstraction, then others.
    """
    FAMILY_PRIORITY = {
        "R_Recombination": 0,
        "H_Abstraction": 1,
        "Decomposition": 2,
        "Disproportionation": 3,
        "R_Addition_MultipleBond": 4,
        "Other": 5,
    }
    sorted_pool = sorted(
        candidate_pool,
        key=lambda r: FAMILY_PRIORITY.get(r.get("family", "Other"), 5)
    )
    result = []
    for r in sorted_pool:
        if r["label"] not in current_labels:
            result.append(r)
        if len(result) >= n:
            break
    return result


def run(state: OrchestratorState, llm) -> SelectorOutput:
    """
    Run the chemistry selector agent.

    Args:
        state: Current orchestrator state (read-only)
        llm:   LLMOrchestrator instance owned by this agent

    Returns:
        SelectorOutput with selected reactions and reasoning
    """
    context        = state.context_for_selector()
    candidate_pool = state.candidate_pool
    current_labels = {r["label"] for r in state.current_reactions}

    # Inject family coverage if available from flux analysis
    if state.flux_analysis.get("family_coverage"):
        context["family_coverage"] = state.flux_analysis["family_coverage"]

    prompt = build_prompt(context, candidate_pool)

    logger.info(
        f"[Selector] Running — pool={len(candidate_pool)}, "
        f"current={len(current_labels)}, n_to_select={context['n_to_select']}"
    )

    raw = llm.chat(
        messages=[{"role": "user", "content": prompt}],
        system_prompt=SYSTEM_PROMPT,
    )

    logger.debug(f"[Selector] Raw output: {raw[:1000]}")

    return parse_and_validate(
        raw,
        candidate_pool,
        current_labels,
        context["n_to_select"],
    )
