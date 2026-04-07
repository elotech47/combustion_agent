"""
DB Retrieval Tool — queries your RMG-crawled species and reactions databases.

Provides:
  - get_species(name)          → adjacency list + metadata
  - get_reactions_for_species  → all reactions involving a species
  - get_reactions_for_fuel     → curated H2/O2 subset (or any fuel list)
  - search_reactions_by_label  → substring search on reaction labels
  - get_library_reactions      → all reactions from a named library (e.g. GRI-Mech3.0)
"""

import json
import logging
import re
from pathlib import Path
from functools import lru_cache

import config

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_reactions_db() -> dict:
    path = Path(config.REACTIONS_DB_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Reactions DB not found: {path.resolve()}")
    logger.info(f"Loading reactions DB from {path} ...")
    with open(path) as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _load_species_db() -> dict:
    path = Path(config.SPECIES_DB_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Species DB not found: {path.resolve()}")
    logger.info(f"Loading species DB from {path} ...")
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Species queries
# ─────────────────────────────────────────────────────────────────────────────

def get_species(name: str) -> dict | None:
    """
    Return species metadata dict or None if not found.
    Searches all label aliases.
    """
    db = _load_species_db()
    name_upper = name.upper()
    for _hash, entry in db.items():
        labels = [l.upper() for l in entry.get("labels", [])]
        if name_upper in labels:
            return {
                "name": name,
                "aliases": entry["labels"],
                "adjacency_list": entry.get("adjacency_list", ""),
                "source_file": entry.get("source_file", ""),
            }
    return None


def list_all_species_names() -> list[str]:
    """Return flat list of all known species labels."""
    db = _load_species_db()
    names = []
    for entry in db.values():
        names.extend(entry.get("labels", []))
    return sorted(set(names))


# ─────────────────────────────────────────────────────────────────────────────
# Reaction queries
# ─────────────────────────────────────────────────────────────────────────────

def get_reactions_for_species(species_name: str, limit: int = 50) -> list[dict]:
    """
    Return all reactions in the DB that involve `species_name`
    on either the reactant or product side.
    """
    db = _load_reactions_db()
    name_upper = species_name.upper()
    results = []
    for rxn in db.get("reactions", []):
        reactants = [r.upper() for r in rxn.get("reactants", [])]
        products  = [p.upper() for p in rxn.get("products",  [])]
        if name_upper in reactants or name_upper in products:
            results.append(_slim_reaction(rxn))
        if len(results) >= limit:
            break
    return results


def get_reactions_for_fuel(
    fuel_species: list[str],
    library_preference: list[str] | None = None,
    limit: int = 200,
) -> list[dict]:
    """
    Return reactions relevant to a fuel mechanism.

    Strategy:
      1. Find all reactions where at least one species matches any species in fuel_species.
      2. De-duplicate using CANONICAL reaction key so that forward/reverse variants of
         the same physical reaction (e.g. 'H2O2 <=> OH + OH' vs 'OH + OH <=> H2O2')
         collapse to one entry. When multiple variants exist, prefer:
           a. Preferred library first
           b. Then best kinetics type: Troe > Lindemann > ThirdBody > Arrhenius
      3. Return up to `limit` results, preferred-library reactions first.
    """
    db = _load_reactions_db()
    fuel_upper = {s.upper() for s in fuel_species}
    preferred  = list(library_preference or [])
    preferred_set = set(preferred)

    matched: list[dict] = []
    for rxn in db.get("reactions", []):
        label = rxn.get("label", "")
        involved = set(_parse_species_from_label(label))
        if involved & fuel_upper:
            matched.append(_slim_reaction(rxn))

    # Rank helpers
    _KINETICS_RANK = {"Troe": 0, "Lindemann": 1, "ThirdBody": 2, "Arrhenius": 3}

    def _lib_rank(r: dict) -> int:
        lib = r.get("library", "")
        try:
            return preferred.index(lib)   # lower index = higher preference
        except ValueError:
            return len(preferred)          # non-preferred libraries go last

    # De-duplicate by CANONICAL key (direction-agnostic).
    # When forward and reverse variants exist, keep the one from the preferred
    # library; break ties by kinetics completeness (Troe wins over Arrhenius).
    seen: dict[str, dict] = {}
    for r in matched:
        key = _canonical_reaction_key(r["label"])
        if key not in seen:
            seen[key] = r
        else:
            existing = seen[key]
            r_lib  = _lib_rank(r)
            ex_lib = _lib_rank(existing)
            r_kt   = _KINETICS_RANK.get(r.get("kinetics_type", ""), 4)
            ex_kt  = _KINETICS_RANK.get(existing.get("kinetics_type", ""), 4)
            # Prefer preferred library; break ties by kinetics rank (lower = better)
            if (r_lib, r_kt) < (ex_lib, ex_kt):
                seen[key] = r

    unique = list(seen.values())

    # Sort: preferred libraries first, then by kinetics rank
    unique.sort(key=lambda r: (_lib_rank(r), _KINETICS_RANK.get(r.get("kinetics_type", ""), 4)))

    return unique[:limit]


def _canonical_reaction_key(label: str) -> str:
    """
    Direction-agnostic canonical key for a reaction label.
    'H2O2 <=> OH + OH' and 'OH + OH <=> H2O2' → same key.
    Strips third-body notation, sorts species on each side, then
    sorts the two sides so the smaller side is always first.
    """
    label = re.sub(r'\s*\(\+?M\)\s*', '', label)
    label = re.sub(r'\s*\+\s*M\b', '', label)
    label = label.strip()
    for sep in ['<=>', '=>', '=']:
        if sep in label:
            parts = label.split(sep, 1)
            lhs = tuple(sorted(s.strip().lower() for s in parts[0].split('+')))
            rhs = tuple(sorted(s.strip().lower() for s in parts[1].split('+')))
            if lhs > rhs:
                lhs, rhs = rhs, lhs
            return f"{'+'.join(lhs)}<=>{'+'.join(rhs)}"
    return label.lower()


def inspect_raw_reactions(limit: int = 5, library: str | None = None) -> dict:
    """
    Return raw (un-slimmed) reaction records from the DB for debugging.
    Shows exact field structure including kinetics format.
    """
    db = _load_reactions_db()
    reactions = db.get("reactions", [])
    if library:
        reactions = [r for r in reactions if r.get("library_name", "") == library]
    samples = reactions[:limit]
    # Return full records so we can see kinetics format exactly
    return {
        "count": len(reactions),
        "sample_records": samples,
        "kinetics_formats_seen": list({
            type(r.get("kinetics", "")).__name__
            for r in reactions[:200]
        }),
        "sample_labels": [r.get("label", "") for r in reactions[:20]],
    }


def search_reactions_by_label(pattern: str, limit: int = 30) -> list[dict]:
    """Substring / regex search on reaction label strings."""
    db = _load_reactions_db()
    regex = re.compile(pattern, re.IGNORECASE)
    results = []
    for rxn in db.get("reactions", []):
        if regex.search(rxn.get("label", "")):
            results.append(_slim_reaction(rxn))
        if len(results) >= limit:
            break
    return results


def get_library_reactions(library_name: str) -> list[dict]:
    """Return all reactions from a specific named library."""
    db = _load_reactions_db()
    lib_reactions = db.get("by_library", {}).get(library_name, [])
    return [_slim_reaction(r) for r in lib_reactions]


def list_available_libraries() -> list[str]:
    """List all library names in the database."""
    db = _load_reactions_db()
    return sorted(db.get("by_library", {}).keys())


def get_db_stats() -> dict:
    """Quick summary of DB contents."""
    rdb = _load_reactions_db()
    sdb = _load_species_db()
    return {
        "total_reactions": rdb.get("total_reactions", len(rdb.get("reactions", []))),
        "libraries": len(rdb.get("by_library", {})),
        "kinetics_types": list(rdb.get("by_kinetics_type", {}).keys()),
        "total_species_entries": len(sdb),
        "species_sample": list_all_species_names()[:20],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _slim_reaction(rxn: dict) -> dict:
    """Return a lighter reaction dict (drop huge raw fields)."""
    return {
        "label":    rxn.get("label", ""),
        "reactants": rxn.get("reactants", []),
        "products":  rxn.get("products",  []),
        "kinetics":  rxn.get("kinetics",  ""),
        "kinetics_type": rxn.get("kinetics_type", ""),
        "library":   rxn.get("library_name", ""),
        "rank":      rxn.get("rank", None),
        "duplicate": rxn.get("duplicate", False),
    }


def _parse_species_from_label(label: str) -> list[str]:
    """
    Extract species names from a reaction label like 'H2 + O <=> OH + H'.
    Strips third-body notation like (M), (Ar) etc.
    """
    label = re.sub(r'\([A-Za-z]+\)', '', label)  # remove (M), (Ar)...
    label = label.replace('<=>', '+').replace('=', '+')
    tokens = [t.strip() for t in label.split('+')]
    return [t for t in tokens if t and not t.startswith('(')]


# ─────────────────────────────────────────────────────────────────────────────
# Tool schema (Anthropic format — LLMOrchestrator converts for other backends)
# ─────────────────────────────────────────────────────────────────────────────

TOOL_SCHEMAS = [
    {
        "name": "get_species_info",
        "description": "Look up a chemical species by name in the RMG database. Returns adjacency list and metadata.",
        "input_schema": {
            "type": "object",
            "properties": {
                "species_name": {"type": "string", "description": "Species name, e.g. H2, OH, CH4"}
            },
            "required": ["species_name"],
        },
    },
    {
        "name": "get_reactions_for_fuel",
        "description": (
            "Retrieve reactions from the RMG database relevant to a given fuel mechanism. "
            "Provide the list of key species in the fuel system. "
            "Optionally specify preferred libraries (e.g. 'GRI-Mech3.0', 'H2-O2')."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "fuel_species": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of species to search for, e.g. ['H2', 'O2', 'H', 'O', 'OH', 'H2O', 'HO2', 'H2O2']"
                },
                "library_preference": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Preferred libraries to draw reactions from first (optional).",
                },
                "limit": {"type": "integer", "description": "Max reactions to return (default 200)"},
            },
            "required": ["fuel_species"],
        },
    },
    {
        "name": "search_reactions_by_label",
        "description": "Search reactions by label pattern (e.g. 'H2O2', 'HO2 <=> OH'). Useful to find specific pathways.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex or substring pattern to match in reaction labels"},
                "limit":   {"type": "integer", "description": "Max results"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "list_available_libraries",
        "description": "List all kinetics libraries available in the database.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "inspect_raw_reactions",
        "description": (
            "DEBUG TOOL: Return raw reaction records from the DB showing exact field structure. "
            "Use this to understand the kinetics format before calling build_mechanism. "
            "Call this if build_mechanism keeps failing due to kinetics parsing errors."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "limit":   {"type": "integer", "description": "Number of records to return (default 5)"},
                "library": {"type": "string",  "description": "Filter to a specific library (optional)"},
            },
            "required": [],
        },
    },



    {
        "name": "get_db_stats",
        "description": "Get summary statistics of the reactions and species databases.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]


def dispatch(tool_name: str, args: dict) -> str:
    try:
        if tool_name == "get_species_info":
            result = get_species(args["species_name"])
            return json.dumps(result or {"error": f"Species '{args['species_name']}' not found"})

        elif tool_name == "get_reactions_for_fuel":
            result = get_reactions_for_fuel(
                args["fuel_species"],
                args.get("library_preference"),
                args.get("limit", 200),
            )
            return json.dumps({"count": len(result), "reactions": result})

        elif tool_name == "search_reactions_by_label":
            result = search_reactions_by_label(args["pattern"], args.get("limit", 30))
            return json.dumps({"count": len(result), "reactions": result})

        elif tool_name == "inspect_raw_reactions":
            result = inspect_raw_reactions(args.get("limit", 5), args.get("library"))
            return json.dumps(result, default=str)

        elif tool_name == "list_available_libraries":
            return json.dumps({"libraries": list_available_libraries()})

        elif tool_name == "get_db_stats":
            return json.dumps(get_db_stats())

        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

    except Exception as e:
        logger.exception(f"DB tool error: {tool_name}")
        return json.dumps({"error": str(e)})