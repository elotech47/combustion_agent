"""
Literature Search Tool.

Gives the agent access to web search for:
  - Reaction rate constants from NIST, literature
  - Experimental ignition delay data (ReSpecTh format)
  - Recent papers on mechanism development for a fuel
  - Reference mechanisms (GRI-Mech, H2/O2 mechanisms)

Uses simple HTTP requests — no API key needed for NIST WebBook.
For general search, uses DuckDuckGo Instant Answer API (free, no key).
"""

import json
import logging
import urllib.parse
import urllib.request

logger = logging.getLogger(__name__)

NIST_KINETICS_BASE = "https://kinetics.nist.gov/kinetics/Search.jsp"
DDG_API = "https://api.duckduckgo.com/"


# ─────────────────────────────────────────────────────────────────────────────
# NIST Kinetics Database search
# ─────────────────────────────────────────────────────────────────────────────

def search_nist_kinetics(reactants: list[str], products: list[str] | None = None) -> dict:
    """
    Search NIST Chemical Kinetics Database for a specific reaction.
    Returns URL and query info — actual parsing requires HTML scraping
    (kept simple for POC; agent can follow up with full fetch if needed).
    """
    query_parts = [f"reactant={urllib.parse.quote(r)}" for r in reactants]
    if products:
        query_parts += [f"product={urllib.parse.quote(p)}" for p in products]

    url = f"{NIST_KINETICS_BASE}?" + "&".join(query_parts)

    return {
        "nist_search_url": url,
        "reactants": reactants,
        "products": products or [],
        "note": (
            "Follow this URL to retrieve Arrhenius parameters from NIST. "
            "The database contains experimentally measured rate constants. "
            "Look for A (pre-exponential), n (temperature exponent), Ea (activation energy)."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# General literature search (DuckDuckGo)
# ─────────────────────────────────────────────────────────────────────────────

def search_literature(query: str, max_results: int = 5) -> dict:
    """
    Search for combustion literature using DuckDuckGo Instant Answer API.
    Returns abstracts and links relevant to the query.
    """
    try:
        params = urllib.parse.urlencode({
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1",
        })
        url     = f"{DDG_API}?{params}"
        req     = urllib.request.Request(url, headers={"User-Agent": "CombustionAgent/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        results = []

        # Abstract (if any)
        if data.get("AbstractText"):
            results.append({
                "title":   data.get("Heading", query),
                "snippet": data["AbstractText"][:500],
                "url":     data.get("AbstractURL", ""),
                "source":  data.get("AbstractSource", ""),
            })

        # Related topics
        for topic in data.get("RelatedTopics", [])[:max_results]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append({
                    "title":   topic.get("Text", "")[:80],
                    "snippet": topic.get("Text", "")[:300],
                    "url":     topic.get("FirstURL", ""),
                    "source":  "DuckDuckGo",
                })

        return {
            "success": True,
            "query": query,
            "results": results[:max_results],
            "note": "Use these results to inform which reactions/rates to add to the mechanism.",
        }

    except Exception as e:
        logger.warning(f"Literature search failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "query": query,
            "results": [],
            "note": "Search failed — agent should proceed with DB-based information only.",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Reference mechanism registry
# ─────────────────────────────────────────────────────────────────────────────

REFERENCE_MECHANISMS = {
    "H2/O2 (Li et al. 2004)": {
        "species":   ["H2", "O2", "H", "O", "OH", "H2O", "HO2", "H2O2", "N2", "AR"],
        "n_reactions": 19,
        "description": "Compact H2/O2 mechanism, widely validated.",
        "url": "https://combustion.ess.uci.edu/H2-O2.html",
        "key_reactions": [
            "H + O2 <=> O + OH",
            "O + H2 <=> H + OH",
            "OH + H2 <=> H + H2O",
            "OH + OH <=> O + H2O",
            "H + H + M <=> H2 + M",
            "H + OH + M <=> H2O + M",
            "H + O2 + M <=> HO2 + M",
            "HO2 + H <=> OH + OH",
            "HO2 + H <=> H2 + O2",
            "HO2 + OH <=> H2O + O2",
            "H2O2 + M <=> OH + OH + M",
            "H2O2 + H <=> HO2 + H2",
            "H2O2 + OH <=> HO2 + H2O",
        ],
    },
    "GRI-Mech 3.0": {
        "species":   325,
        "n_reactions": 325,
        "description": "Full natural gas mechanism. Overkill for pure H2 but good reference.",
        "url": "http://combustion.berkeley.edu/gri-mech/",
    },
}


def get_reference_mechanism_info(fuel: str = "H2") -> dict:
    """
    Return info about published reference mechanisms for a given fuel.
    Agent uses this to compare its generated mechanism against known baselines.
    """
    relevant = {}
    fuel_upper = fuel.upper()
    for name, info in REFERENCE_MECHANISMS.items():
        if fuel_upper in name.upper() or fuel_upper in str(info.get("species", [])):
            relevant[name] = info

    return {
        "fuel": fuel,
        "reference_mechanisms": relevant,
        "recommendation": (
            "Compare your generated mechanism's IDT predictions against "
            "the Li et al. 2004 H2/O2 mechanism as ground truth."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tool schemas + dispatcher
# ─────────────────────────────────────────────────────────────────────────────

TOOL_SCHEMAS = [
    {
        "name": "search_literature",
        "description": (
            "Search combustion literature for information about a reaction, species, "
            "or mechanism. Useful for finding rate constant estimates, experimental "
            "validation data, or recent mechanism development work."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query, e.g. 'H2 O2 ignition delay mechanism Li 2004' or 'HO2 + OH rate constant'",
                },
                "max_results": {"type": "integer", "description": "Max results to return"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_nist_kinetics",
        "description": (
            "Search the NIST Chemical Kinetics Database for Arrhenius parameters "
            "for a specific reaction. Returns the search URL and guidance."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reactants": {"type": "array", "items": {"type": "string"}, "description": "Reactant species names"},
                "products":  {"type": "array", "items": {"type": "string"}, "description": "Product species names (optional)"},
            },
            "required": ["reactants"],
        },
    },
    {
        "name": "get_reference_mechanism_info",
        "description": "Get information about published reference mechanisms for a given fuel (H2, CH4, etc.).",
        "input_schema": {
            "type": "object",
            "properties": {
                "fuel": {"type": "string", "description": "Fuel name, e.g. H2, CH4, C2H4"},
            },
            "required": ["fuel"],
        },
    },
]


def dispatch(tool_name: str, args: dict) -> str:
    try:
        if tool_name == "search_literature":
            return json.dumps(search_literature(args["query"], args.get("max_results", 5)))
        elif tool_name == "search_nist_kinetics":
            return json.dumps(search_nist_kinetics(args["reactants"], args.get("products")))
        elif tool_name == "get_reference_mechanism_info":
            return json.dumps(get_reference_mechanism_info(args.get("fuel", "H2")))
        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
    except Exception as e:
        logger.exception(f"Literature tool error: {tool_name}")
        return json.dumps({"error": str(e)})
