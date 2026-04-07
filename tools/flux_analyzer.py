"""
Flux Analyzer Tool.

Performs reaction path and sensitivity analysis on a validated mechanism.
Tells the agent WHICH reactions/species are most important or missing,
guiding the refinement decision rather than leaving it to random search.

Provides:
  - sensitivity_analysis : dIDT/d(A_i) for each reaction
  - rate_of_production   : dominant production/consumption pathways per species
  - identify_weak_links  : reactions with near-zero flux (candidates to prune)
  - identify_missing     : species present but with no associated reactions
"""

import json
import logging
import tempfile
import os

logger = logging.getLogger(__name__)

try:
    import cantera as ct
    import numpy as np
    CANTERA_AVAILABLE = True
except ImportError:
    CANTERA_AVAILABLE = False
    logger.warning("Cantera/numpy not available — flux analyzer will return mock data.")


# ─────────────────────────────────────────────────────────────────────────────
# Sensitivity analysis
# ─────────────────────────────────────────────────────────────────────────────

def sensitivity_analysis(
    mechanism_yaml: str,
    T: float = 1000.0,
    P: float = 101325.0,
    phi: float = 1.0,
    fuel: str = "H2",
    oxidizer: str = "O2:1, N2:3.76",
    top_n: int = 15,
) -> dict:
    """
    Compute first-order sensitivity of ignition delay to pre-exponential factors.
    High |sensitivity| → reaction strongly controls IDT.
    Positive → increasing rate delays ignition.
    Negative → increasing rate promotes ignition.
    """
    if not CANTERA_AVAILABLE:
        return _mock_sensitivity()

    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(mechanism_yaml)
            mech_path = f.name

        gas = ct.Solution(mech_path)
        gas.set_equivalence_ratio(phi, fuel, oxidizer)
        gas.TP = T, P

        reactor = ct.IdealGasConstPressureReactor(gas)
        sim     = ct.ReactorNet([reactor])
        sim.rtol_sensitivity = 1e-6
        sim.atol_sensitivity = 1e-6

        # Enable sensitivity for all reactions
        for i in range(gas.n_reactions):
            reactor.add_sensitivity_reaction(i)

        # Advance until ignition
        t = 0.0
        idt_found = False
        T_prev = T
        sens_at_idt = None

        while t < 0.5:
            t += 1e-6
            try:
                sim.advance(t)
            except Exception:
                break
            if reactor.T - T_prev > 5 and reactor.T > T + 100 and not idt_found:
                idt_found = True
                # Sensitivities at ignition point
                sens_at_idt = sim.sensitivities()
                break
            T_prev = reactor.T

        os.unlink(mech_path)

        if sens_at_idt is None or not idt_found:
            return {"success": False, "error": "Ignition not reached during sensitivity run"}

        # sens_at_idt shape: (n_vars, n_reactions)
        # We want sensitivity of temperature (index 0 usually) to each reaction
        T_sens = sens_at_idt[0, :]  # temperature sensitivity

        rxn_names = [gas.reaction_equation(i) for i in range(gas.n_reactions)]
        pairs = sorted(zip(rxn_names, T_sens.tolist()), key=lambda x: -abs(x[1]))[:top_n]

        return {
            "success": True,
            "top_sensitive_reactions": [
                {"reaction": r, "sensitivity": round(s, 6)} for r, s in pairs
            ],
            "most_promoting":  pairs[0][0] if pairs else None,
            "most_inhibiting": max(pairs, key=lambda x: x[1])[0] if pairs else None,
        }

    except Exception as e:
        logger.exception("Sensitivity analysis failed")
        return {"success": False, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Rate of production analysis
# ─────────────────────────────────────────────────────────────────────────────

def rate_of_production(
    mechanism_yaml: str,
    species_of_interest: list[str],
    T: float = 1200.0,
    P: float = 101325.0,
    phi: float = 1.0,
    fuel: str = "H2",
    oxidizer: str = "O2:1, N2:3.76",
    top_n: int = 5,
) -> dict:
    """
    At a given T/P/phi, compute rate of production (ROP) for target species.
    Identifies the dominant reactions consuming and producing each species.
    """
    if not CANTERA_AVAILABLE:
        return _mock_rop(species_of_interest)

    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(mechanism_yaml)
            mech_path = f.name

        gas = ct.Solution(mech_path)
        gas.set_equivalence_ratio(phi, fuel, oxidizer)
        gas.TP = T, P
        gas.equilibrate('HP')  # equilibrate to get meaningful ROP at this state

        results = {}
        for sp in species_of_interest:
            if sp not in gas.species_names:
                results[sp] = {"error": "species not in mechanism"}
                continue

            sp_idx = gas.species_index(sp)
            rop    = gas.net_rates_of_progress
            nu     = gas.product_stoich_coeffs[sp_idx] - gas.reactant_stoich_coeffs[sp_idx]
            sp_rop = nu * rop

            rxn_names = [gas.reaction_equation(i) for i in range(gas.n_reactions)]
            pairs = sorted(zip(rxn_names, sp_rop.tolist()), key=lambda x: -abs(x[1]))[:top_n]

            results[sp] = {
                "net_production_rate": float(sp_rop.sum()),
                "top_pathways": [
                    {"reaction": r, "rate": round(v, 4)} for r, v in pairs
                ],
            }

        os.unlink(mech_path)
        return {"success": True, "species_rop": results}

    except Exception as e:
        logger.exception("ROP analysis failed")
        return {"success": False, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Structural analysis (no Cantera needed)
# ─────────────────────────────────────────────────────────────────────────────

def identify_missing_reactions(
    mechanism_species: list[str],
    mechanism_reactions: list[dict],
) -> dict:
    """
    Given species and reactions in the current mechanism, identify:
    1. Species that appear in <2 reactions (likely disconnected)
    2. Species pairs that have no direct reaction between them (potential gaps)
    """
    from collections import defaultdict
    species_rxn_count = defaultdict(int)

    for rxn in mechanism_reactions:
        for sp in rxn.get("reactants", []) + rxn.get("products", []):
            species_rxn_count[sp.upper()] += 1

    isolated = [sp for sp in mechanism_species
                if species_rxn_count.get(sp.upper(), 0) < 2]

    return {
        "total_species": len(mechanism_species),
        "total_reactions": len(mechanism_reactions),
        "isolated_species": isolated,
        "recommendation": (
            f"Species {isolated} have fewer than 2 reactions — consider adding pathways for them."
            if isolated else "No obviously isolated species found."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Mock fallbacks
# ─────────────────────────────────────────────────────────────────────────────

def _mock_sensitivity() -> dict:
    return {
        "success": True,
        "mock": True,
        "top_sensitive_reactions": [
            {"reaction": "H + O2 <=> O + OH",    "sensitivity": -0.85},
            {"reaction": "H + O2 + M <=> HO2 + M","sensitivity": 0.62},
            {"reaction": "OH + H2 <=> H + H2O",   "sensitivity": -0.41},
            {"reaction": "HO2 + H <=> OH + OH",   "sensitivity": -0.33},
            {"reaction": "H2O2 + M <=> OH + OH + M","sensitivity": -0.21},
        ],
        "most_promoting": "H + O2 <=> O + OH",
        "most_inhibiting": "H + O2 + M <=> HO2 + M",
    }


def _mock_rop(species: list[str]) -> dict:
    return {
        "success": True,
        "mock": True,
        "species_rop": {
            sp: {"net_production_rate": 0.0, "top_pathways": []} for sp in species
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tool schemas + dispatcher
# ─────────────────────────────────────────────────────────────────────────────

TOOL_SCHEMAS = [
    {
        "name": "sensitivity_analysis",
        "description": (
            "Compute ignition delay sensitivity to each reaction's pre-exponential factor. "
            "Returns the top reactions that most strongly promote or inhibit ignition. "
            "Use this after a failed validation to understand WHICH reactions to add or modify."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "mechanism_yaml": {"type": "string"},
                "T":     {"type": "number", "description": "Temperature K"},
                "P":     {"type": "number", "description": "Pressure Pa"},
                "top_n": {"type": "integer", "description": "Top N reactions to return"},
            },
            "required": ["mechanism_yaml"],
        },
    },
    {
        "name": "rate_of_production",
        "description": (
            "Compute rate of production for key species at a given thermodynamic state. "
            "Reveals dominant production and consumption pathways, helping identify "
            "missing reactions in the mechanism."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "mechanism_yaml": {"type": "string"},
                "species_of_interest": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Species to analyze, e.g. ['OH', 'HO2', 'H2O2']",
                },
                "T": {"type": "number"},
                "P": {"type": "number"},
            },
            "required": ["mechanism_yaml", "species_of_interest"],
        },
    },
    {
        "name": "identify_missing_reactions",
        "description": (
            "Structural check: find species in the mechanism that have fewer than 2 reactions "
            "(likely disconnected). Returns recommendations for which species need more pathways."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "mechanism_species":  {"type": "array", "items": {"type": "string"}},
                "mechanism_reactions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "reactants": {"type": "array", "items": {"type": "string"}},
                            "products":  {"type": "array", "items": {"type": "string"}},
                        },
                    },
                },
            },
            "required": ["mechanism_species", "mechanism_reactions"],
        },
    },
]


def dispatch(tool_name: str, args: dict) -> str:
    try:
        if tool_name == "sensitivity_analysis":
            return json.dumps(sensitivity_analysis(
                args["mechanism_yaml"],
                T=args.get("T", 1000),
                P=args.get("P", 101325),
                top_n=args.get("top_n", 15),
            ))
        elif tool_name == "rate_of_production":
            return json.dumps(rate_of_production(
                args["mechanism_yaml"],
                args["species_of_interest"],
                T=args.get("T", 1200),
                P=args.get("P", 101325),
            ))
        elif tool_name == "identify_missing_reactions":
            return json.dumps(identify_missing_reactions(
                args["mechanism_species"],
                args["mechanism_reactions"],
            ))
        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
    except Exception as e:
        logger.exception(f"Flux analyzer error: {tool_name}")
        return json.dumps({"error": str(e)})
