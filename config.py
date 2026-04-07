"""
Configuration for the Combustion Mechanism Agent.
Set your preferred backend and API keys here (or via environment variables).
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM Backend ────────────────────────────────────────────────────────────────
# Options: "claude" | "openai" | "openrouter" | "local"
LLM_BACKEND = os.getenv("LLM_BACKEND", "claude")

# ── Model names per backend ────────────────────────────────────────────────────
MODELS = {
    "claude":      os.getenv("CLAUDE_MODEL",      "claude-opus-4-5"),
    "openai":      os.getenv("OPENAI_MODEL",       "gpt-4o"),
    "openrouter":  os.getenv("OPENROUTER_MODEL",   "anthropic/claude-opus-4-5"),
    "local":       os.getenv("LOCAL_MODEL",        "qwen2.5:72b"),  # or whatever you're serving
}

# ── API Keys ───────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY  = os.getenv("ANTHROPIC_API_KEY",  "")
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY",     "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Local inference server (vLLM / Ollama / LM Studio)
LOCAL_BASE_URL = os.getenv("LOCAL_BASE_URL", "http://localhost:11434/v1")
LOCAL_API_KEY  = os.getenv("LOCAL_API_KEY",  "ollama")  # dummy for Ollama

# ── Database paths ─────────────────────────────────────────────────────────────
REACTIONS_DB_PATH = os.getenv("REACTIONS_DB_PATH", "database/reactions_database.json")
SPECIES_DB_PATH   = os.getenv("SPECIES_DB_PATH",   "database/species_database.json")
FAMILIES_DB_PATH  = os.getenv("FAMILIES_DB_PATH",  "database/families_database.json")

# ── Agent settings ─────────────────────────────────────────────────────────────
MAX_REFINEMENT_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "5"))

# ── Fuel configurations ─────────────────────────────────────────────────────────
# Each entry defines the starting point for a given fuel.
# Add entries here as thermo data is added to mechanism_builder.NASA7_DATA / _COMPOSITIONS.
FUEL_CONFIGS: dict[str, dict] = {
    "H2": {
        # Species to seed the initial DB search with
        "initial_species": ["H2", "O2", "H", "O", "OH", "H2O", "HO2", "H2O2"],
        # RMG library search preference order.
        # Specialist H2/O2 libraries come FIRST — they store reactions in the
        # physically correct direction (e.g. H2O2 <=> OH + OH for decomposition)
        # and have been validated against H2/O2 experiments.
        "library_preference": [
            "primaryH2O2",
            "BurkeH2O2inArHe",
            "BurkeH2O2inN2",
            "FFCM1(-)",
            "Glarborg/C0",
            "GRI-Mech3.0",
            "JetSurF2.0",
        ],
        # Bath gases / diluents to always include in the mechanism
        "bath_gases": ["N2", "AR"],
        # Primary oxidizer species
        "oxidizer": "O2",
    },
    "CH4": {
        "initial_species": ["CH4", "O2", "H", "O", "OH", "H2O", "CO", "CO2",
                             "HCO", "CH3", "CH2O", "H2", "HO2"],
        "library_preference": ["GRI-Mech3.0", "JetSurF2.0", "combustion"],
        "bath_gases": ["N2", "AR"],
        "oxidizer": "O2",
    },
    "C2H6": {
        "initial_species": ["C2H6", "O2", "H", "O", "OH", "H2O", "CO", "CO2",
                             "HCO", "CH3", "C2H5", "C2H4", "H2", "HO2"],
        "library_preference": ["GRI-Mech3.0", "JetSurF2.0"],
        "bath_gases": ["N2", "AR"],
        "oxidizer": "O2",
    },
}

def get_fuel_config(fuel: str) -> dict:
    """Return fuel config, falling back to a generic template if unknown."""
    key = fuel.upper()
    for k in FUEL_CONFIGS:
        if k.upper() == key:
            return FUEL_CONFIGS[k]
    # Generic fallback: use the fuel itself as the only initial species
    return {
        "initial_species": [fuel.upper(), "O2", "H", "O", "OH", "H2O"],
        "library_preference": ["GRI-Mech3.0", "JetSurF2.0", "combustion"],
        "bath_gases": ["N2", "AR"],
        "oxidizer": "O2",
    }

# Validation targets for H2/O2 @ phi=1.0, P=1atm, T=1000K
# These are reference values the agent tries to match
VALIDATION_TARGETS = {
    "ignition_delay_ms": {
        "T1000_P1atm": 0.15,   # ms — approximate for H2/O2
        "T1200_P1atm": 0.02,
        "T800_P1atm":  2.5,
    },
    "tolerance_pct": 30.0,     # accept if within 30% of target
}

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
SAVE_INTERMEDIATE_MECHANISMS = True
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
