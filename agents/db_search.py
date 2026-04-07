"""
DB Search Agent.

Pure retrieval — no LLM involved.
Given target species and already-tried labels, queries the RMG database
and returns a fresh candidate pool.

This is intentionally deterministic: same inputs → same outputs.
The intelligence lives in the selector, not here.

RMG-inspired design:
  RMG generates reactions by applying family templates to ALL PAIRS of core
  species. We mirror this by doing two queries:
    1. Broad query: reactions involving any target species (fuel + diagnostician targets)
    2. Core-pair query: reactions where BOTH reactants are already in the mechanism
       — this ensures established species get complete chemistry coverage
"""

import json
import logging
from collections import Counter
from agents.state import OrchestratorState
from tools import db_retrieval, family_tool

logger = logging.getLogger(__name__)


def run(state: OrchestratorState) -> list[dict]:
    """
    Query the RMG database for reactions relevant to the current target species.

    Returns:
        List of candidate reaction dicts (slim format from db_retrieval)
    """
    context = state.context_for_db_search()

    target_species = context["target_species"]
    already_tried  = set(context["already_tried_labels"])
    current_labels = set(context["current_reaction_labels"])
    library_pref   = context["library_preference"]
    core_species   = [s.upper() for s in state.current_species]  # species in current mechanism

    excluded = already_tried | current_labels

    # Build the set of species that are acceptable for this fuel system.
    # A reaction is only valid if ALL its species are in this set.
    # This prevents carbon/nitrogen/etc. reactions from contaminating an H2/O2 run.
    # Bath gases (N2, AR, HE) and the abstract third-body token M are always allowed.
    _ALWAYS_ALLOWED = {"N2", "AR", "HE", "M", "N2(S)"}
    fuel_species_upper = {s.upper() for s in target_species}
    core_species_upper = {s.upper() for s in state.current_species}
    bath_gases_upper   = {s.upper() for s in state.bath_gases}
    allowed_species = (
        fuel_species_upper | core_species_upper | bath_gases_upper | _ALWAYS_ALLOWED
    )

    def _all_species_allowed(rxn: dict) -> bool:
        """Return True only if every species in the reaction is in the allowed set."""
        involved = set(rxn.get("reactants", [])) | set(rxn.get("products", []))
        return all(s.upper() in allowed_species for s in involved)

    logger.info(
        f"[DB Search] target_species={target_species}, "
        f"already_tried={len(already_tried)}, library_pref={library_pref[:3]}"
    )

    # ── Query 1: Broad search across all target species ────────────────────────
    result_str = db_retrieval.dispatch("get_reactions_for_fuel", {
        "fuel_species":       target_species,
        "library_preference": library_pref,
        "limit":              300,
    })
    result = json.loads(result_str)
    all_reactions = result.get("reactions", [])
    logger.info(f"[DB Search] Broad query: {len(all_reactions)} reactions")

    seen_canonical = set()
    candidates = []
    for r in all_reactions:
        if r["label"] in excluded:
            continue
        if not _all_species_allowed(r):
            continue
        ck = db_retrieval._canonical_reaction_key(r["label"])
        if ck not in seen_canonical:
            seen_canonical.add(ck)
            candidates.append(r)

    # ── Query 2: Core-pair targeted search (RMG-inspired) ──────────────────────
    # For each species already in the mechanism, fetch ALL its reactions.
    # This ensures that when a new species is added, its full chemistry is visible.
    # Prioritises reactions where both reactants are established species.
    if core_species:
        for sp in core_species:
            sp_str = db_retrieval.dispatch("get_reactions_for_fuel", {
                "fuel_species":       [sp] + core_species,
                "library_preference": library_pref,
                "limit":              50,
            })
            sp_result = json.loads(sp_str)
            for r in sp_result.get("reactions", []):
                if r["label"] in excluded:
                    continue
                if not _all_species_allowed(r):
                    continue
                ck = db_retrieval._canonical_reaction_key(r["label"])
                if ck not in seen_canonical:
                    seen_canonical.add(ck)
                    candidates.append(r)

    # ── Query 3: Targeted search for diagnostician-requested pathways ───────────
    # Only search for pathways that involve species already in the allowed set.
    # This prevents the diagnostician from pulling in carbon chemistry by naming
    # carbon species as "missing" after seeing them in a Cantera error message.
    if state.diagnostician_report and state.diagnostician_report.missing_pathways:
        for pathway in state.diagnostician_report.missing_pathways[:5]:
            # Extract species tokens from the pathway string
            tokens = [t.strip() for t in
                      pathway.replace('<=>', '+').replace('=>', '+').replace('=', '+').split('+')
                      if t.strip() and t.strip().upper() not in ('M', '')]
            # Skip this pathway if any of its species are outside the fuel system
            if any(t.upper() not in allowed_species for t in tokens):
                logger.debug(f"[DB Search] Skipping out-of-scope pathway: {pathway}")
                continue
            if not tokens:
                continue
            # Search for each key species from the pathway
            for token in tokens[:2]:   # use first two species as search keys
                pat_str = db_retrieval.dispatch("get_reactions_for_fuel", {
                    "fuel_species":       [token] + (core_species or target_species),
                    "library_preference": library_pref,
                    "limit":              30,
                })
                pat_result = json.loads(pat_str)
                for r in pat_result.get("reactions", []):
                    if r["label"] in excluded:
                        continue
                    if not _all_species_allowed(r):
                        continue
                    ck = db_retrieval._canonical_reaction_key(r["label"])
                    if ck not in seen_canonical:
                        seen_canonical.add(ck)
                        candidates.append(r)

    # ── Query 4: Ensure third-body coverage on iteration 0 ────────────────────
    tb_in_pool = sum(1 for r in candidates if _is_third_body(r))
    if tb_in_pool < 3 and state.iteration == 0:
        logger.info("[DB Search] Augmenting with explicit third-body search")
        for pattern in [r"\+ M\b", r"\(\+M\)"]:
            tb_str = db_retrieval.dispatch("search_reactions_by_label", {
                "pattern": pattern,
                "limit":   50,
            })
            tb_result = json.loads(tb_str)
            for r in tb_result.get("reactions", []):
                if r["label"] in excluded:
                    continue
                if not _all_species_allowed(r):
                    continue
                ck = db_retrieval._canonical_reaction_key(r["label"])
                if ck not in seen_canonical:
                    seen_canonical.add(ck)
                    candidates.append(r)

    # ── Classify all candidates by reaction family (RMG-style) ──────────────
    # This adds a 'family' field to each reaction dict so the selector can
    # reason about chain roles (termination, propagation, etc.) rather than
    # just individual reaction labels.
    candidates = family_tool.classify_reactions_by_family(candidates)

    family_counts = Counter(r.get("family", "Other") for r in candidates)
    logger.info(f"[DB Search] Candidate pool: {len(candidates)} reactions | "
                f"families={dict(family_counts.most_common(6))}")

    sample = [r["label"] for r in candidates[:10]]
    logger.debug(f"[DB Search] Sample candidates: {sample}")

    return candidates


def _is_third_body(rxn: dict) -> bool:
    """Use kinetics_type as authoritative signal; fall back to label string."""
    kt = rxn.get("kinetics_type", "")
    if kt in ("ThirdBody", "Troe", "Lindemann", "PDepArrhenius", "MultiPDepArrhenius"):
        return True
    label_up = rxn.get("label", "").upper()
    return "(+M)" in label_up or "+ M " in label_up or label_up.strip().endswith("+M")
