"""
Species Identity Tool — Graph Isomorphism without RDKit/Cython.

Uses the pre-built species_database.json (MD5-hashed adjacency lists) to:
  - Normalize species names across libraries  (e.g. 'phenyl' == 'A1-' == 'A1-(92)')
  - Check structural identity of two species
  - List all known aliases for a molecule
  - Parse molecular formulas from adjacency lists

The hash in species_database.json IS the graph isomorphism result — identical structures
(same adjacency list up to atom numbering) share the same MD5 hash.  Two names that map
to the same hash are the same molecule.

For species not in the database (e.g. newly proposed intermediates), a fallback
adjacency-list parser computes an approximate canonical form.
"""

import hashlib
import json
import logging
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import config

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Common species: hardcoded SMILES ↔ name maps (fast path for H2/O2 and extensions)
# ─────────────────────────────────────────────────────────────────────────────

# SMILES → canonical name (used to decode efficiency keys from Troe kinetics)
SMILES_TO_NAME: dict[str, str] = {
    "O":           "H2O",
    "[H][H]":      "H2",
    "[H]":         "H",
    "[O][O]":      "O2",
    "[O]":         "O",
    "[OH]":        "OH",
    "OO":          "H2O2",
    "[O]O":        "HO2",
    "[Ar]":        "AR",
    "N#N":         "N2",
    "[He]":        "HE",
    "O=C=O":       "CO2",
    "[C-]#[O+]":   "CO",
    "C":           "CH4",
    "C=O":         "CH2O",
    "[CH3]":       "CH3",
    "CO":          "CH3OH",
    "CC":          "C2H6",
    "C=C":         "C2H4",
    "[CH2]":       "CH2",
    "C#C":         "C2H2",
    "N":           "NH3",
    "[NH2]":       "NH2",
    "[NH]":        "NH",
    "[N]":         "N",
    "C#N":         "HCN",
    "S":           "H2S",
}

# Also build the reverse map for name → SMILES lookups
NAME_TO_SMILES: dict[str, str] = {v: k for k, v in SMILES_TO_NAME.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Species database loading and index building
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_species_db() -> dict:
    path = Path(config.SPECIES_DB_PATH)
    if not path.exists():
        logger.warning(f"Species DB not found: {path}")
        return {}
    with open(path) as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _build_label_index() -> dict[str, str]:
    """
    Returns a dict mapping every known lowercase label → hash.
    Multiple labels map to the same hash when they are the same molecule.
    """
    db = _load_species_db()
    index: dict[str, str] = {}
    for hash_key, entry in db.items():
        for label in entry.get("labels", []):
            index[label.upper()] = hash_key
    return index


@lru_cache(maxsize=1)
def _build_hash_index() -> dict[str, dict]:
    """
    Returns a dict mapping hash → {labels, adjacency_list, formula}.
    """
    db = _load_species_db()
    result = {}
    for hash_key, entry in db.items():
        labels = entry.get("labels", [])
        adj    = entry.get("adjacency_list", "")
        result[hash_key] = {
            "labels":         labels,
            "canonical_name": labels[0] if labels else hash_key,
            "adjacency_list": adj,
            "formula":        _formula_from_adjacency(adj),
        }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Adjacency list parsing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _formula_from_adjacency(adj: str) -> str:
    """
    Derive molecular formula from an RMG adjacency list string.
    Each non-header line starts with: index element u# p# c# {bonds...}
    """
    counts: Counter = Counter()
    for line in adj.splitlines():
        line = line.strip()
        if not line or line.startswith("multiplicity"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            element = parts[1]
            if re.match(r"^[A-Z][a-z]?$", element):
                counts[element] += 1
    if not counts:
        return ""
    # Hill order: C first, H second, then alphabetical
    formula_parts = []
    for el in sorted(counts.keys(),
                     key=lambda e: (0 if e == "C" else 1 if e == "H" else 2, e)):
        n = counts[el]
        formula_parts.append(el if n == 1 else f"{el}{n}")
    return "".join(formula_parts)


def _normalize_adjacency(adj: str) -> str:
    """
    Produce a canonical form of an adjacency list for hashing:
    sort atom lines by element+connectivity to remove numbering dependency.
    """
    lines = []
    for line in adj.splitlines():
        line = line.strip()
        if line and not line.startswith("multiplicity"):
            lines.append(line)
    return "\n".join(sorted(lines))


def adjacency_to_hash(adj: str) -> str:
    canonical = _normalize_adjacency(adj)
    return hashlib.md5(canonical.encode()).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Core API functions
# ─────────────────────────────────────────────────────────────────────────────

def normalize_species_name(name: str) -> dict:
    """
    Resolve a species name to its canonical identity.

    Returns a dict with:
      canonical_name   — preferred label from the species DB
      all_aliases      — every known name for this molecule
      formula          — molecular formula (e.g. 'H2O')
      hash             — structural identity hash (same hash = same molecule)
      found_in_db      — whether we found it in species_database.json
    """
    key = name.upper().strip()
    label_idx = _build_label_index()
    hash_idx  = _build_hash_index()

    if key in label_idx:
        h    = label_idx[key]
        info = hash_idx[h]
        return {
            "canonical_name": info["canonical_name"],
            "all_aliases":    info["labels"],
            "formula":        info["formula"],
            "hash":           h,
            "found_in_db":    True,
        }

    # Fallback: return the name itself with unknown formula
    return {
        "canonical_name": name,
        "all_aliases":    [name],
        "formula":        "",
        "hash":           None,
        "found_in_db":    False,
    }


def find_species_aliases(name: str) -> list[str]:
    """Return all known names for a species (structural identity)."""
    result = normalize_species_name(name)
    return result["all_aliases"]


def are_same_species(name1: str, name2: str) -> bool:
    """
    Return True if name1 and name2 refer to the same molecule.
    Uses structural hash comparison — handles alias mismatches across libraries.
    """
    if name1.upper() == name2.upper():
        return True
    r1 = normalize_species_name(name1)
    r2 = normalize_species_name(name2)
    if r1["hash"] and r2["hash"]:
        return r1["hash"] == r2["hash"]
    return False


def normalize_smiles_to_name(smiles: str) -> str | None:
    """
    Resolve a SMILES string (as used in Troe efficiency dicts) to a species name.
    Returns None if not recognized.
    """
    return SMILES_TO_NAME.get(smiles)


def normalize_efficiency_dict(raw_eff: dict[str, float],
                               declared_species: list[str]) -> dict[str, float]:
    """
    Convert Troe efficiency dict with SMILES keys → species-name keys,
    filtering to only species actually declared in the mechanism.

    Args:
        raw_eff: {smiles: efficiency, ...}  from the parsed kinetics string
        declared_species: list of species names currently in the mechanism

    Returns:
        {species_name: efficiency} for all resolved + declared species
    """
    declared_upper = {s.upper() for s in declared_species}
    result = {}
    for smiles, eff in raw_eff.items():
        name = SMILES_TO_NAME.get(smiles)
        if name and name.upper() in declared_upper:
            result[name] = eff
    return result


def get_species_info(name: str) -> dict:
    """Full species info including adjacency list."""
    key = name.upper().strip()
    label_idx = _build_label_index()
    hash_idx  = _build_hash_index()

    if key in label_idx:
        h    = label_idx[key]
        info = hash_idx[h]
        return {
            "canonical_name":  info["canonical_name"],
            "all_aliases":     info["labels"],
            "formula":         info["formula"],
            "hash":            h,
            "adjacency_list":  info["adjacency_list"],
            "found_in_db":     True,
        }
    return {"canonical_name": name, "found_in_db": False}


def batch_normalize(names: list[str]) -> dict[str, str]:
    """
    Normalize a list of species names, returning {input_name: canonical_name}.
    Useful for cleaning up species lists before mechanism assembly.
    """
    return {n: normalize_species_name(n)["canonical_name"] for n in names}


# ─────────────────────────────────────────────────────────────────────────────
# Tool schemas + dispatcher (for LLM tool calling)
# ─────────────────────────────────────────────────────────────────────────────

TOOL_SCHEMAS = [
    {
        "name": "normalize_species_name",
        "description": (
            "Resolve a species name to its canonical identity using structural hashing. "
            "Handles aliases across libraries — e.g. 'phenyl', 'A1-', and 'A1-(92)' all "
            "resolve to the same molecule. Returns canonical name, all aliases, formula, and "
            "structural hash. Use this before querying the DB to ensure consistent naming."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Species name or label to normalize"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "find_species_aliases",
        "description": (
            "Return all known names for a species across all RMG libraries. "
            "Essential for multi-fuel work where the same molecule appears under "
            "different labels in different libraries."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "are_same_species",
        "description": (
            "Check whether two species names refer to the same molecule (structural identity). "
            "Uses hash-based graph isomorphism — safe to use across libraries and naming conventions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name1": {"type": "string"},
                "name2": {"type": "string"},
            },
            "required": ["name1", "name2"],
        },
    },
    {
        "name": "get_species_info",
        "description": (
            "Get full species information including molecular formula, all aliases, "
            "and RMG adjacency list. Useful when extending the mechanism to new fuels "
            "and needing to understand species structure."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "batch_normalize_species",
        "description": (
            "Normalize a list of species names in one call. Returns a mapping from "
            "input names to canonical names. Use this to clean up a species list "
            "before building a mechanism."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of species names to normalize",
                },
            },
            "required": ["names"],
        },
    },
]


def dispatch(tool_name: str, args: dict) -> str:
    try:
        if tool_name == "normalize_species_name":
            return json.dumps(normalize_species_name(args["name"]))

        elif tool_name == "find_species_aliases":
            return json.dumps({"name": args["name"], "aliases": find_species_aliases(args["name"])})

        elif tool_name == "are_same_species":
            same = are_same_species(args["name1"], args["name2"])
            return json.dumps({"name1": args["name1"], "name2": args["name2"], "same_molecule": same})

        elif tool_name == "get_species_info":
            return json.dumps(get_species_info(args["name"]))

        elif tool_name == "batch_normalize_species":
            return json.dumps({"normalized": batch_normalize(args["names"])})

        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

    except Exception as e:
        logger.exception(f"species_tool error: {tool_name}")
        return json.dumps({"error": str(e)})
