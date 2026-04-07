"""
Reaction Family Tool — RMG-inspired family-aware reaction search and coverage analysis.

In real RMG (step 2 of the workflow), candidate reactions are enumerated by applying
reaction family templates (H_Abstraction, R_Recombination, Disproportionation, etc.)
to all core species via graph isomorphism.

This tool approximates that by:
  1. Classifying every reaction in the DB into a family using structural heuristics
  2. Providing a coverage checker: given a mechanism, which families are missing?
  3. Providing family-guided search: fetch reactions from a specific family
  4. Listing families relevant to a given species set

This lets the LLM selector reason about COMPLETENESS ("have I covered chain termination
via R_Recombination?") rather than just individual reactions.
"""

import json
import logging
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Optional

import config

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Structural heuristics for reaction family classification
# ─────────────────────────────────────────────────────────────────────────────

# Radical species — used to detect recombination/abstraction patterns
_RADICAL_SPECIES = {
    "H", "O", "OH", "HO2", "CH3", "NH2", "NH", "N", "CN", "HCO",
    "CH2", "CH", "C2H", "C2H3", "C2H5", "CH2OH", "CH3O",
}

# Species that contain transferable H atoms (for H_Abstraction detection)
_H_DONORS = {"H2", "H2O", "H2O2", "HO2", "CH4", "C2H6", "NH3", "H2S", "H2CO"}


def _parse_species(label: str) -> tuple[list[str], list[str]]:
    """Split a reaction label into reactant and product species lists."""
    label = re.sub(r'\s*\(\+?M\)\s*', '', label)
    label = re.sub(r'\s*\+\s*M\b', '', label)
    for sep in ['<=>', '=>', '=']:
        if sep in label:
            lhs, rhs = label.split(sep, 1)
            r_sp = [s.strip() for s in lhs.split('+') if s.strip()]
            p_sp = [s.strip() for s in rhs.split('+') if s.strip()]
            return r_sp, p_sp
    return [], []


def classify_reaction(
    label: str,
    kinetics_type: str = "",
    reactants: Optional[list[str]] = None,
    products: Optional[list[str]] = None,
) -> str:
    """
    Classify a reaction into an RMG-style reaction family.

    Returns one of:
      H_Abstraction, R_Recombination, Disproportionation, Decomposition,
      R_Addition_MultipleBond, Intra_H_migration, Other
    """
    if reactants is None or products is None:
        reactants, products = _parse_species(label)

    r_clean = [s.upper() for s in reactants if s.upper() not in ("M", "(M)", "(+M)")]
    p_clean = [s.upper() for s in products  if s.upper() not in ("M", "(M)", "(+M)")]
    n_r, n_p = len(r_clean), len(p_clean)

    is_tb = kinetics_type in ("ThirdBody", "Troe", "Lindemann")

    # ── Recombination: A• + B• → AB (+M optional) ───────────────────────────
    if n_r == 2 and n_p == 1 and (is_tb or kinetics_type in ("Arrhenius", "")):
        r_set = {s.upper() for s in r_clean}
        if r_set & {s.upper() for s in _RADICAL_SPECIES}:
            return "R_Recombination"

    # ── Decomposition: AB (+M) → A + B ──────────────────────────────────────
    if n_r == 1 and n_p == 2:
        return "Decomposition"
    if n_r == 1 and n_p >= 2:
        return "Decomposition"

    # ── H_Abstraction: A-H + B• → A• + B-H ─────────────────────────────────
    if n_r == 2 and n_p == 2:
        # Check if exactly one H atom is transferred
        r_upper = {s.upper() for s in r_clean}
        p_upper = {s.upper() for s in p_clean}
        r_h_donors = r_upper & {s.upper() for s in _H_DONORS}
        r_radicals  = r_upper & {s.upper() for s in _RADICAL_SPECIES}
        if r_h_donors and r_radicals:
            return "H_Abstraction"
        # Broader check: any H-containing + any radical
        if r_radicals and any(
            re.search(r'H', s, re.I) for s in r_clean if s.upper() not in _RADICAL_SPECIES
        ):
            return "H_Abstraction"

    # ── Disproportionation: two radicals → two non-radical products ─────────
    if n_r == 2 and n_p == 2:
        r_upper = {s.upper() for s in r_clean}
        if len(r_upper & {s.upper() for s in _RADICAL_SPECIES}) >= 1:
            return "Disproportionation"

    # ── R_Addition_MultipleBond: radical + unsaturated → adduct ─────────────
    if n_r == 2 and n_p == 1 and not is_tb:
        return "R_Addition_MultipleBond"

    # ── Intra_H_migration: isomerization ───────────────────────────────────
    if n_r == 1 and n_p == 1:
        return "Intra_H_migration"

    return "Other"


# ─────────────────────────────────────────────────────────────────────────────
# Families database
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_families_db() -> dict:
    path = Path(config.FAMILIES_DB_PATH)
    if not path.exists():
        logger.warning(f"Families DB not found: {path}")
        return {"families": {}}
    with open(path) as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _load_reactions_db() -> list[dict]:
    path = Path(config.REACTIONS_DB_PATH)
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    return data.get("reactions", [])


# ─────────────────────────────────────────────────────────────────────────────
# Core API functions
# ─────────────────────────────────────────────────────────────────────────────

def list_reaction_families(relevant_species: Optional[list[str]] = None) -> dict:
    """
    List all known reaction families with descriptions.
    If relevant_species is given, filters to families likely to apply.

    Returns a dict with:
      families: [{name, template_reactants, template_products, rate_rules_count, description}]
      n_total: total number of families
    """
    fdb = _load_families_db()
    families = fdb.get("families", {})

    result = []
    for name, info in families.items():
        entry = {
            "name":              name,
            "template_reactants": info.get("template_reactants", []),
            "template_products":  info.get("template_products", []),
            "rate_rules_count":   info.get("rate_rules_count", 0),
            "training_count":     info.get("training_reactions_count", 0),
            "examples":           info.get("example_reactions", [])[:2],
        }
        result.append(entry)

    # For H/O chemistry, highlight the most relevant families
    CORE_FAMILIES = {
        "H_Abstraction", "R_Recombination", "Disproportionation",
        "Decomposition", "R_Addition_MultipleBond", "HO2_Elimination_from_PeroxyRadical",
        "Intra_H_migration",
    }
    result.sort(key=lambda x: (0 if x["name"] in CORE_FAMILIES else 1, x["name"]))

    return {"families": result, "n_total": len(result)}


def classify_reactions_by_family(reaction_dicts: list[dict]) -> list[dict]:
    """
    Classify a list of reaction dicts into families.

    Input: list of {label, kinetics_type, reactants, products, ...}
    Returns same list with 'family' field added to each.
    """
    result = []
    for rxn in reaction_dicts:
        family = classify_reaction(
            rxn.get("label", ""),
            rxn.get("kinetics_type", ""),
            rxn.get("reactants"),
            rxn.get("products"),
        )
        enriched = dict(rxn)
        enriched["family"] = family
        result.append(enriched)
    return result


def check_mechanism_coverage(
    species_list: list[str],
    reaction_labels: list[str],
    kinetics_types: Optional[list[str]] = None,
) -> dict:
    """
    Analyse which reaction families are covered in the current mechanism
    and which are expected but missing.

    For a mechanism with given species, returns:
      covered_families:   families that have ≥1 reaction in mechanism
      missing_families:   expected families with 0 reactions
      family_breakdown:   {family: [reaction_labels]}
      recommendations:    human-readable list of what to add next

    This is the agent's equivalent of RMG step 7 (checking coverage before promotion).
    """
    if kinetics_types is None:
        kinetics_types = ["Arrhenius"] * len(reaction_labels)

    # Classify current mechanism reactions
    family_breakdown: dict[str, list[str]] = defaultdict(list)
    for label, ktype in zip(reaction_labels, kinetics_types):
        reactants, products = _parse_species(label)
        family = classify_reaction(label, ktype, reactants, products)
        family_breakdown[family].append(label)

    covered = set(family_breakdown.keys())

    # Determine expected families for the given species
    species_upper = {s.upper() for s in species_list}
    expected_families = set()

    has_radicals = bool(species_upper & {s.upper() for s in _RADICAL_SPECIES})
    has_h_donors  = bool(species_upper & {s.upper() for s in _H_DONORS})
    has_peroxy    = bool(species_upper & {"HO2", "H2O2", "CH3O2"})

    if has_radicals:
        expected_families.add("R_Recombination")
    if has_radicals and has_h_donors:
        expected_families.add("H_Abstraction")
    if has_peroxy:
        expected_families.add("Decomposition")
        expected_families.add("Disproportionation")
    if len(species_upper) >= 3:
        expected_families.add("R_Addition_MultipleBond")

    missing = expected_families - covered

    # Build recommendations
    recommendations = []
    if "R_Recombination" in missing:
        recommendations.append(
            "R_Recombination missing: add ThirdBody/Troe reactions for radical+radical → stable "
            "(e.g. H+OH+M→H2O+M, H+O2+M→HO2+M)"
        )
    if "H_Abstraction" in missing:
        recommendations.append(
            "H_Abstraction missing: add reactions like OH+H2→H+H2O, H+H2O→OH+H2"
        )
    if "Decomposition" in missing:
        recommendations.append(
            "Decomposition missing: add H2O2+M→OH+OH+M or HO2 decomposition"
        )
    if "Disproportionation" in missing:
        recommendations.append(
            "Disproportionation missing: add HO2+HO2→H2O2+O2, HO2+OH→H2O+O2"
        )

    return {
        "covered_families":  sorted(covered),
        "missing_families":  sorted(missing),
        "expected_families": sorted(expected_families),
        "family_breakdown":  {k: v for k, v in family_breakdown.items()},
        "recommendations":   recommendations,
        "coverage_pct":      round(
            100 * len(covered & expected_families) / max(len(expected_families), 1), 1
        ),
    }


def get_reactions_by_family(
    family_name: str,
    species_list: list[str],
    limit: int = 30,
) -> dict:
    """
    Fetch reactions from the DB that belong to a given family and
    involve only species in species_list.

    This is the family-guided equivalent of get_reactions_for_fuel() —
    it narrows the search by chemical mechanism family rather than just
    species name matching.
    """
    reactions = _load_reactions_db()
    species_upper = {s.upper() for s in species_list}

    results = []
    for rxn in reactions:
        # Species filter
        involved = {s.upper() for s in rxn.get("reactants", []) + rxn.get("products", [])}
        if not involved.issubset(species_upper):
            continue

        # Family filter
        fam = classify_reaction(
            rxn.get("label", ""),
            rxn.get("kinetics_type", ""),
            rxn.get("reactants"),
            rxn.get("products"),
        )
        if fam.lower() == family_name.lower():
            results.append({
                "label":        rxn["label"],
                "kinetics_type": rxn.get("kinetics_type", ""),
                "library":      rxn.get("library_name", ""),
                "family":       fam,
            })
        if len(results) >= limit:
            break

    return {
        "family":     family_name,
        "n_found":    len(results),
        "reactions":  results,
    }


def get_family_for_reaction(label: str, kinetics_type: str = "",
                             reactants: Optional[list[str]] = None,
                             products: Optional[list[str]] = None) -> str:
    """Return the reaction family name for a single reaction."""
    return classify_reaction(label, kinetics_type, reactants, products)


def get_completeness_report(
    species_list: list[str],
    mechanism_reactions: list[dict],
) -> dict:
    """
    Full completeness report: family coverage + gaps + priority recommendations.
    This is the primary tool the diagnostician should call after partial success.

    Equivalent to RMG steps 6-7: flux analysis → promotion/pruning decision.
    """
    labels    = [r.get("label", "") for r in mechanism_reactions]
    ktypes    = [r.get("kinetics_type", "") for r in mechanism_reactions]
    coverage  = check_mechanism_coverage(species_list, labels, ktypes)
    classified = classify_reactions_by_family(mechanism_reactions)
    family_counts = defaultdict(int)
    for r in classified:
        family_counts[r["family"]] += 1

    return {
        **coverage,
        "reaction_count_by_family": dict(family_counts),
        "total_reactions": len(mechanism_reactions),
        "total_species":   len(species_list),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tool schemas + dispatcher
# ─────────────────────────────────────────────────────────────────────────────

TOOL_SCHEMAS = [
    {
        "name": "list_reaction_families",
        "description": (
            "List all known RMG reaction families with their templates and descriptions. "
            "Use this at the start of a new fuel system to understand which reaction classes "
            "are relevant. Core families for H2/O2: H_Abstraction, R_Recombination, "
            "Disproportionation, Decomposition, R_Addition_MultipleBond."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "relevant_species": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "If provided, filter to families applicable to these species",
                },
            },
            "required": [],
        },
    },
    {
        "name": "check_mechanism_coverage",
        "description": (
            "Analyse which reaction families are covered by the current mechanism and which "
            "are expected but missing. Returns covered_families, missing_families, and "
            "human-readable recommendations. Use this after partial validation success to "
            "identify structural gaps — equivalent to RMG's promotion/pruning step."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "species_list": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "All species in the current mechanism",
                },
                "reaction_labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Labels of all reactions in the current mechanism",
                },
                "kinetics_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Kinetics type of each reaction (parallel to reaction_labels)",
                },
            },
            "required": ["species_list", "reaction_labels"],
        },
    },
    {
        "name": "get_reactions_by_family",
        "description": (
            "Fetch candidate reactions from the DB that belong to a specific family and "
            "involve only the given species. More targeted than species-name search — use "
            "this when the coverage check shows a family is missing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "family_name": {
                    "type": "string",
                    "description": "Family name, e.g. 'H_Abstraction', 'R_Recombination'",
                },
                "species_list": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Species to search within",
                },
                "limit": {"type": "integer", "description": "Max results (default 30)"},
            },
            "required": ["family_name", "species_list"],
        },
    },
    {
        "name": "get_completeness_report",
        "description": (
            "Full family coverage report for the current mechanism. Returns which families "
            "are covered, which are missing, reaction counts per family, and priority "
            "recommendations. This is the primary diagnostic tool for structural completeness — "
            "call it when score is partial or when the diagnostician needs guidance."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "species_list": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "mechanism_reactions": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "List of reaction dicts with at least 'label' and 'kinetics_type'",
                },
            },
            "required": ["species_list", "mechanism_reactions"],
        },
    },
    {
        "name": "classify_reaction",
        "description": "Classify a single reaction into its RMG family.",
        "input_schema": {
            "type": "object",
            "properties": {
                "label":         {"type": "string"},
                "kinetics_type": {"type": "string", "default": ""},
            },
            "required": ["label"],
        },
    },
]


def dispatch(tool_name: str, args: dict) -> str:
    try:
        if tool_name == "list_reaction_families":
            return json.dumps(list_reaction_families(args.get("relevant_species")))

        elif tool_name == "check_mechanism_coverage":
            return json.dumps(check_mechanism_coverage(
                args["species_list"],
                args["reaction_labels"],
                args.get("kinetics_types"),
            ))

        elif tool_name == "get_reactions_by_family":
            return json.dumps(get_reactions_by_family(
                args["family_name"],
                args["species_list"],
                args.get("limit", 30),
            ))

        elif tool_name == "get_completeness_report":
            return json.dumps(get_completeness_report(
                args["species_list"],
                args["mechanism_reactions"],
            ))

        elif tool_name == "classify_reaction":
            family = classify_reaction(args["label"], args.get("kinetics_type", ""))
            return json.dumps({"label": args["label"], "family": family})

        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

    except Exception as e:
        logger.exception(f"family_tool error: {tool_name}")
        return json.dumps({"error": str(e)})
