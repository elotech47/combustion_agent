"""
Cantera Validation Tool.

Given a mechanism (as a YAML string or file path), computes:
  - Ignition delay time (IDT) via 0D constant-pressure reactor
  - Laminar flame speed (LFS) via 1D freely propagating flame
  - Major species profiles at ignition

Falls back gracefully if cantera is not installed (returns mock data for dev/testing).
"""

import json
import logging
import tempfile
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Try importing cantera — graceful fallback for environments without it
try:
    import cantera as ct
    CANTERA_AVAILABLE = True
    logger.info(f"Cantera {ct.__version__} loaded.")
except ImportError:
    CANTERA_AVAILABLE = False
    logger.warning("Cantera not available — validation tool will return mock results.")


# ─────────────────────────────────────────────────────────────────────────────
# Core validation functions
# ─────────────────────────────────────────────────────────────────────────────

def compute_ignition_delay(
    mechanism_yaml: str,
    T: float = 1000.0,
    P: float = 101325.0,
    phi: float = 1.0,
    fuel: str = "H2",
    oxidizer: str = "O2:1, N2:3.76",
    max_time: float = 1.0,
    n_steps: int = 10000,
) -> dict:
    """
    Compute ignition delay time using a 0D constant-pressure reactor.

    Args:
        mechanism_yaml: Cantera YAML mechanism as a string
        T: Initial temperature (K)
        P: Pressure (Pa)
        phi: Equivalence ratio
        fuel: Fuel species string
        oxidizer: Oxidizer string (Cantera format)
        max_time: Max integration time (s)
        n_steps: Number of time steps

    Returns:
        dict with 'ignition_delay_ms', 'peak_T', 'time_series', 'success', 'error'
    """
    if not CANTERA_AVAILABLE:
        return _mock_idt(T, P, phi)

    try:
        # Write mechanism to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(mechanism_yaml)
            mech_path = f.name

        gas = ct.Solution(mech_path)
        gas.set_equivalence_ratio(phi, fuel, oxidizer)
        gas.TP = T, P

        reactor  = ct.IdealGasConstPressureReactor(gas)
        sim      = ct.ReactorNet([reactor])

        dt       = max_time / n_steps
        times    = []
        temps    = []
        t        = 0.0
        T_prev   = T
        idt      = None

        for _ in range(n_steps):
            t += dt
            sim.advance(t)
            T_now = reactor.T
            times.append(t * 1000)   # ms
            temps.append(T_now)

            # IDT defined as time of max dT/dt
            if T_now - T_prev > 5.0 and idt is None and T_now > T + 100:
                idt = t * 1000  # ms

            T_prev = T_now
            if T_now > 2800:
                break

        os.unlink(mech_path)

        return {
            "success": True,
            "ignition_delay_ms": round(idt, 4) if idt else None,
            "peak_T": round(max(temps), 1),
            "T_initial": T,
            "P_Pa": P,
            "phi": phi,
            "time_series_ms": times[::max(1, len(times)//50)],  # downsample
            "temp_series_K":  temps[::max(1, len(temps)//50)],
        }

    except Exception as e:
        logger.exception("IDT computation failed")
        return {"success": False, "error": str(e), "ignition_delay_ms": None}


def compute_laminar_flame_speed(
    mechanism_yaml: str,
    T: float = 300.0,
    P: float = 101325.0,
    phi: float = 1.0,
    fuel: str = "H2",
    oxidizer: str = "O2:1, N2:3.76",
) -> dict:
    """
    Compute laminar flame speed using 1D freely propagating flame.

    Returns dict with 'lfs_cm_s', 'success', 'error'.
    """
    if not CANTERA_AVAILABLE:
        return _mock_lfs(T, P, phi)

    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(mechanism_yaml)
            mech_path = f.name

        gas = ct.Solution(mech_path)
        gas.set_equivalence_ratio(phi, fuel, oxidizer)
        gas.TP = T, P

        flame = ct.FreeFlame(gas, width=0.03)
        flame.set_refine_criteria(ratio=3, slope=0.06, curve=0.12)
        flame.solve(loglevel=0, auto=True)

        lfs = flame.velocity[0] * 100  # m/s → cm/s
        os.unlink(mech_path)

        return {
            "success": True,
            "lfs_cm_s": round(lfs, 2),
            "T_initial": T,
            "P_Pa": P,
            "phi": phi,
        }

    except Exception as e:
        logger.exception("LFS computation failed")
        return {"success": False, "error": str(e), "lfs_cm_s": None}


def validate_mechanism(
    mechanism_yaml: str,
    conditions: list[dict] | None = None,
    compute_lfs: bool = True,
) -> dict:
    """
    Full validation pass: runs IDT across multiple T/P conditions + LFS.

    Default conditions are chosen to be representative for H2/O2.

    Returns a scored validation report the agent uses to decide whether to refine.
    """
    if conditions is None:
        conditions = [
            {"T": 800,  "P": 101325, "phi": 1.0, "label": "T800_P1atm"},
            {"T": 1000, "P": 101325, "phi": 1.0, "label": "T1000_P1atm"},
            {"T": 1200, "P": 101325, "phi": 1.0, "label": "T1200_P1atm"},
            {"T": 1000, "P": 506625, "phi": 1.0, "label": "T1000_P5atm"},
        ]

    idt_results = []
    for cond in conditions:
        res = compute_ignition_delay(
            mechanism_yaml,
            T=cond["T"],
            P=cond["P"],
            phi=cond.get("phi", 1.0),
        )
        res["condition_label"] = cond["label"]
        idt_results.append(res)

    lfs_result = None
    if compute_lfs:
        lfs_result = compute_laminar_flame_speed(mechanism_yaml)

    # Score: fraction of conditions that successfully ignited
    n_success = sum(1 for r in idt_results if r.get("success") and r.get("ignition_delay_ms"))
    score = n_success / len(idt_results) if idt_results else 0.0

    return {
        "validation_score": round(score, 3),
        "n_conditions": len(idt_results),
        "n_successful": n_success,
        "idt_results": idt_results,
        "lfs_result": lfs_result,
        "passed": score >= 0.75,
        "summary": _summarize_validation(idt_results, lfs_result, score),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Mock fallbacks (when Cantera not available)
# ─────────────────────────────────────────────────────────────────────────────

def _mock_idt(T, P, phi) -> dict:
    """Approximate IDT using Arrhenius scaling — for dev without Cantera."""
    import math
    Ea_R = 15000  # K, approximate activation energy / R for H2
    A    = 1e-5
    idt  = A * math.exp(Ea_R / T) * (101325 / P)
    return {
        "success": True,
        "ignition_delay_ms": round(idt * 1000, 4),
        "peak_T": T + 1200,
        "T_initial": T,
        "P_Pa": P,
        "phi": phi,
        "mock": True,
    }


def _mock_lfs(T, P, phi) -> dict:
    return {
        "success": True,
        "lfs_cm_s": 210.0,  # approximate H2/air LFS at phi=1
        "T_initial": T,
        "P_Pa": P,
        "phi": phi,
        "mock": True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _summarize_validation(idt_results, lfs_result, score) -> str:
    lines = [f"Validation score: {score:.1%}"]
    for r in idt_results:
        label = r.get("condition_label", "?")
        idt   = r.get("ignition_delay_ms")
        if r.get("success") and idt:
            lines.append(f"  {label}: IDT = {idt:.3f} ms, peak T = {r.get('peak_T', '?')} K")
        else:
            lines.append(f"  {label}: FAILED — {r.get('error', 'no ignition')}")
    if lfs_result:
        if lfs_result.get("success"):
            lines.append(f"  LFS = {lfs_result.get('lfs_cm_s', '?')} cm/s")
        else:
            lines.append(f"  LFS: FAILED — {lfs_result.get('error', '')}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Tool schema + dispatcher
# ─────────────────────────────────────────────────────────────────────────────

TOOL_SCHEMAS = [
    {
        "name": "validate_mechanism",
        "description": (
            "Validate a combustion mechanism by running Cantera simulations. "
            "Computes ignition delay times across temperature/pressure conditions "
            "and optionally laminar flame speed. Returns a validation score and "
            "detailed results the agent uses to decide whether to refine the mechanism."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "mechanism_yaml": {
                    "type": "string",
                    "description": "Full Cantera YAML mechanism string to validate",
                },
                "compute_lfs": {
                    "type": "boolean",
                    "description": "Whether to also compute laminar flame speed (slower)",
                },
            },
            "required": ["mechanism_yaml"],
        },
    },
    {
        "name": "compute_ignition_delay",
        "description": "Compute ignition delay time for specific T/P/phi conditions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "mechanism_yaml": {"type": "string"},
                "T":   {"type": "number", "description": "Temperature in K"},
                "P":   {"type": "number", "description": "Pressure in Pa"},
                "phi": {"type": "number", "description": "Equivalence ratio"},
            },
            "required": ["mechanism_yaml", "T", "P"],
        },
    },
]


def dispatch(tool_name: str, args: dict) -> str:
    try:
        if tool_name == "validate_mechanism":
            result = validate_mechanism(
                args["mechanism_yaml"],
                compute_lfs=args.get("compute_lfs", False),
            )
            return json.dumps(result)

        elif tool_name == "compute_ignition_delay":
            result = compute_ignition_delay(
                args["mechanism_yaml"],
                T=args.get("T", 1000),
                P=args.get("P", 101325),
                phi=args.get("phi", 1.0),
            )
            return json.dumps(result)

        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

    except Exception as e:
        logger.exception(f"Cantera tool error: {tool_name}")
        return json.dumps({"error": str(e)})
