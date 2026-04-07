"""
Mechanism Builder.

Takes a list of reactions (from the DB) and assembles a valid Cantera YAML
mechanism file. Also handles:
  - Deduplication of reactions
  - NASA 7-coefficient polynomial thermodynamics (GRI-Mech 3.0 / NIST-JANAF)
  - Third-body efficiency assignment
  - Graceful skipping of reactions whose species have no thermo data
"""

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# NASA 7-coefficient thermodynamic data
# Source: GRI-Mech 3.0 / NIST-JANAF
# Format per entry: Tmin, Tmid, Tmax, hi (7 coeffs, Tmid→Tmax), lo (7 coeffs, Tmin→Tmid)
# Extend this dict to support additional fuels — mechanism_builder will automatically
# accept any reaction whose species all appear here.
# ─────────────────────────────────────────────────────────────────────────────

NASA7_DATA = {
    # ── H / O / H2 / O2 core ─────────────────────────────────────────────────
    "H2": {
        "Tmin": 200, "Tmid": 1000, "Tmax": 3500,
        "hi": [3.33727920e+00, -4.94024731e-05,  4.99456778e-07, -1.79566394e-10,  2.00255376e-14, -9.50158922e+02, -3.20502331e+00],
        "lo": [2.34433112e+00,  7.98052075e-03, -1.94781510e-05,  2.01572094e-08, -7.37611761e-12, -9.17935173e+02,  6.83010238e-01],
    },
    "O2": {
        "Tmin": 200, "Tmid": 1000, "Tmax": 3500,
        "hi": [3.69757819e+00,  6.13519689e-04, -1.25884199e-07,  1.77528148e-11, -1.13643531e-15, -1.23393018e+03,  3.18916591e+00],
        "lo": [3.78245636e+00, -2.99673416e-03,  9.84730201e-06, -9.68129509e-09,  3.24372837e-12, -1.06394356e+03,  3.65767573e+00],
    },
    "H": {
        "Tmin": 200, "Tmid": 1000, "Tmax": 3500,
        "hi": [2.50000001e+00, -2.30842973e-11,  1.61561948e-14, -4.73515235e-18,  4.98197357e-22,  2.54736599e+04, -4.46682914e-01],
        "lo": [2.50000000e+00,  7.05332819e-13, -1.99591964e-15,  2.30081632e-18, -9.27732332e-22,  2.54736599e+04, -4.46682853e-01],
    },
    "O": {
        "Tmin": 200, "Tmid": 1000, "Tmax": 3500,
        "hi": [2.56942078e+00, -8.59741137e-05,  4.19484589e-08, -1.00177799e-11,  1.22833691e-15,  2.92175791e+04,  4.78433864e+00],
        "lo": [3.16826710e+00, -3.27931884e-03,  6.64306396e-06, -6.12806624e-09,  2.11265971e-12,  2.91222592e+04,  2.05193346e+00],
    },
    "OH": {
        "Tmin": 200, "Tmid": 1000, "Tmax": 3500,
        "hi": [3.09288767e+00,  5.48429716e-04,  1.26505228e-07, -8.79461556e-11,  1.17412376e-14,  3.85865700e+03,  4.47669610e+00],
        "lo": [3.99201543e+00, -2.40131752e-03,  4.61793841e-06, -3.88113333e-09,  1.36411470e-12,  3.61508056e+03, -1.03925458e-01],
    },
    "H2O": {
        "Tmin": 200, "Tmid": 1000, "Tmax": 3500,
        "hi": [3.03399249e+00,  2.17691804e-03, -1.64072518e-07, -9.70419870e-11,  1.68200992e-14, -3.00042971e+04,  4.96675241e+00],
        "lo": [4.19864056e+00, -2.03643410e-03,  6.52040211e-06, -5.48797062e-09,  1.77197250e-12, -3.02937267e+04, -8.49032208e-01],
    },
    "HO2": {
        "Tmin": 200, "Tmid": 1000, "Tmax": 3500,
        "hi": [4.01721090e+00,  2.23982013e-03, -6.33658150e-07,  1.14246370e-10, -1.07908535e-14,  1.11856713e+02,  3.78510215e+00],
        "lo": [4.30179801e+00, -4.74912051e-03,  2.11582891e-05, -2.42763894e-08,  9.29225124e-12,  2.94808040e+02,  3.71666245e+00],
    },
    "H2O2": {
        "Tmin": 200, "Tmid": 1000, "Tmax": 3500,
        "hi": [4.16500285e+00,  4.90831694e-03, -1.90139225e-06,  3.71185986e-10, -2.87908305e-14, -1.78617877e+04,  2.91615662e+00],
        "lo": [4.27611269e+00, -5.42822417e-04,  1.67335701e-05, -2.15770813e-08,  8.62454363e-12, -1.77025821e+04,  3.43505074e+00],
    },
    "N2": {
        "Tmin": 200, "Tmid": 1000, "Tmax": 3500,
        "hi": [2.95257637e+00,  1.39690040e-03, -4.92631603e-07,  7.86010195e-11, -4.60755204e-15, -9.23948688e+02,  5.87188762e+00],
        "lo": [3.53100528e+00, -1.23660988e-04, -5.02999433e-07,  2.43530612e-09, -1.40881235e-12, -1.04697628e+03,  2.96747038e+00],
    },
    "AR": {
        "Tmin": 200, "Tmid": 1000, "Tmax": 3500,
        "hi": [2.50000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, -7.45375000e+02,  4.37967491e+00],
        "lo": [2.50000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, -7.45375000e+02,  4.37967491e+00],
    },
    # ── C1 species (methane sub-mechanism) ────────────────────────────────────
    "CO": {
        "Tmin": 200, "Tmid": 1000, "Tmax": 3500,
        "hi": [2.71518561e+00,  2.06252743e-03, -9.98825771e-07,  2.30053008e-10, -2.03647716e-14, -1.41518724e+04,  7.81868772e+00],
        "lo": [3.57953347e+00, -6.10353680e-04,  1.01681433e-06,  9.07005884e-10, -9.04424499e-13, -1.43440860e+04,  3.50840928e+00],
    },
    "CO2": {
        "Tmin": 200, "Tmid": 1000, "Tmax": 3500,
        "hi": [3.85746029e+00,  4.41437026e-03, -2.21481404e-06,  5.23490188e-10, -4.72084164e-14, -4.87591660e+04,  2.27163806e+00],
        "lo": [2.35677352e+00,  8.98459677e-03, -7.12356269e-06,  2.45919022e-09, -1.43699548e-13, -4.83719697e+04,  9.90105222e+00],
    },
    "CH4": {
        "Tmin": 200, "Tmid": 1000, "Tmax": 3500,
        "hi": [7.48514950e-02,  1.33909467e-02, -5.73285809e-06,  1.22292535e-09, -1.01815230e-13, -9.46834459e+03,  1.84373180e+01],
        "lo": [5.14987613e+00, -1.36709788e-02,  4.91800599e-05, -4.84743026e-08,  1.66693956e-11, -1.02466476e+04, -4.64130376e+00],
    },
    "CH3": {
        "Tmin": 200, "Tmid": 1000, "Tmax": 3500,
        "hi": [2.28571772e+00,  7.23990037e-03, -2.98714348e-06,  5.95684644e-10, -4.67154394e-14,  1.67755843e+04,  8.48007179e+00],
        "lo": [3.67359040e+00,  2.01095175e-03,  5.73021856e-06, -6.87117425e-09,  2.54385734e-12,  1.64449988e+04,  1.60456433e+00],
    },
    "CH2O": {
        "Tmin": 200, "Tmid": 1000, "Tmax": 3500,
        "hi": [1.76069008e+00,  9.20000082e-03, -4.42258813e-06,  1.00641212e-09, -8.83855640e-14, -1.39958323e+04,  1.36563230e+01],
        "lo": [4.79372315e+00, -9.90833369e-03,  3.73220008e-05, -3.79285261e-08,  1.31772652e-11, -1.43089567e+04,  6.02812430e-01],
    },
    "HCO": {
        "Tmin": 200, "Tmid": 1000, "Tmax": 3500,
        "hi": [2.77217438e+00,  4.95695526e-03, -2.48445613e-06,  5.89161778e-10, -5.33508711e-14,  4.01191815e+03,  9.79834492e+00],
        "lo": [4.22118584e+00, -3.24392532e-03,  1.37799446e-05, -1.33144093e-08,  4.33768865e-12,  3.83956496e+03,  3.39437243e+00],
    },
    "CH2": {
        "Tmin": 200, "Tmid": 1000, "Tmax": 3500,
        "hi": [2.87410113e+00,  3.65639292e-03, -1.40894597e-06,  2.60179549e-10, -1.87727567e-14,  4.62636040e+04,  6.17119324e+00],
        "lo": [3.76267867e+00,  9.68872143e-04,  2.79489752e-06, -3.85091153e-09,  1.68741719e-12,  4.60040401e+04,  1.56253185e+00],
    },
    "CH2(S)": {
        "Tmin": 200, "Tmid": 1000, "Tmax": 3500,
        "hi": [2.29203842e+00,  4.65588637e-03, -2.01191947e-06,  4.17906000e-10, -3.39716365e-14,  5.09259997e+04,  8.62650169e+00],
        "lo": [4.19860411e+00, -2.36661419e-03,  8.23296220e-06, -6.68815981e-09,  1.94314737e-12,  5.04968163e+04, -7.78375430e-01],
    },
    "CH": {
        "Tmin": 200, "Tmid": 1000, "Tmax": 3500,
        "hi": [2.87846473e+00,  9.70913681e-04,  1.44445655e-07, -1.30687849e-10,  1.76079383e-14,  7.10124364e+04,  5.48497999e+00],
        "lo": [3.48981665e+00,  3.23835541e-04, -1.68899065e-06,  3.16217327e-09, -1.40609067e-12,  7.07972934e+04,  2.08401108e+00],
    },
    # ── C2 species (ethane/ethylene sub-mechanism) ────────────────────────────
    "C2H6": {
        "Tmin": 200, "Tmid": 1000, "Tmax": 3500,
        "hi": [1.07188150e+00,  2.16852677e-02, -1.00256067e-05,  2.21412001e-09, -1.90002890e-13, -1.14263932e+04,  1.51156107e+01],
        "lo": [4.29142492e+00, -5.50154270e-03,  5.99438288e-05, -7.08466285e-08,  2.68685771e-11, -1.15222055e+04,  2.66682316e+00],
    },
    "C2H5": {
        "Tmin": 200, "Tmid": 1000, "Tmax": 3500,
        "hi": [1.95465642e+00,  1.73972722e-02, -7.98206668e-06,  1.75217689e-09, -1.49641576e-13,  1.28575200e+04,  1.34624343e+01],
        "lo": [4.30646568e+00, -4.18658892e-03,  4.97142807e-05, -5.99126606e-08,  2.30509004e-11,  1.28416265e+04,  4.70720924e+00],
    },
    "C2H4": {
        "Tmin": 200, "Tmid": 1000, "Tmax": 3500,
        "hi": [2.03611116e+00,  1.46454151e-02, -6.71077915e-06,  1.47222923e-09, -1.25706061e-13,  4.93988614e+03,  1.03053693e+01],
        "lo": [3.95920148e+00, -7.57052247e-03,  5.70990292e-05, -6.91588753e-08,  2.69884373e-11,  5.08977593e+03,  4.09733096e+00],
    },
    "C2H3": {
        "Tmin": 200, "Tmid": 1000, "Tmax": 3500,
        "hi": [3.01672400e+00,  1.03302292e-02, -4.68082349e-06,  1.01763288e-09, -8.62607041e-14,  3.46128739e+04,  7.78732378e+00],
        "lo": [3.21246645e+00,  1.51479162e-03,  2.59209412e-05, -3.57657847e-08,  1.47150873e-11,  3.48598468e+04,  8.51054025e+00],
    },
    "C2H2": {
        "Tmin": 200, "Tmid": 1000, "Tmax": 3500,
        "hi": [4.14756964e+00,  5.96166664e-03, -2.37294852e-06,  4.67412171e-10, -3.61235213e-14,  2.59359992e+04, -1.23028121e+00],
        "lo": [8.08681094e-01,  2.33615629e-02, -3.55171815e-05,  2.80152437e-08, -8.50072974e-12,  2.64289807e+04,  1.39397051e+01],
    },
    "C2H": {
        "Tmin": 200, "Tmid": 1000, "Tmax": 3500,
        "hi": [3.16780652e+00,  4.75221902e-03, -1.83787077e-06,  3.04190252e-10, -1.77232770e-14,  6.71210650e+04,  6.63589475e+00],
        "lo": [2.88965733e+00,  1.34099611e-02, -2.84769501e-05,  2.94791045e-08, -1.09331511e-11,  6.68393932e+04,  6.22296438e+00],
    },
    # ── He (common bath gas) ──────────────────────────────────────────────────
    "HE": {
        "Tmin": 200, "Tmid": 1000, "Tmax": 6000,
        "hi": [2.50000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, -7.45375000e+02,  9.28723974e-01],
        "lo": [2.50000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, -7.45375000e+02,  9.28723974e-01],
    },
}

# Molecular weights (g/mol)
MOLECULAR_WEIGHTS = {
    "H2": 2.016,   "O2": 31.998,  "H": 1.008,    "O": 15.999,
    "OH": 17.007,  "H2O": 18.015, "HO2": 33.006, "H2O2": 34.014,
    "N2": 28.014,  "AR": 39.948,  "HE": 4.003,
    "CO": 28.010,  "CO2": 44.010,
    "CH4": 16.043, "CH3": 15.035, "CH2O": 30.026, "HCO": 29.018,
    "CH2": 14.027, "CH2(S)": 14.027, "CH": 13.019,
    "C2H6": 30.069, "C2H5": 29.062, "C2H4": 28.054,
    "C2H3": 27.046, "C2H2": 26.038, "C2H": 25.030,
}

# Element compositions for Cantera YAML.
# Keys match NASA7_DATA keys (uppercase). Extend in parallel with NASA7_DATA.
_COMPOSITIONS: dict[str, dict[str, int]] = {
    "H2":    {"H": 2},
    "O2":    {"O": 2},
    "H":     {"H": 1},
    "O":     {"O": 1},
    "OH":    {"H": 1, "O": 1},
    "H2O":   {"H": 2, "O": 1},
    "HO2":   {"H": 1, "O": 2},
    "H2O2":  {"H": 2, "O": 2},
    "N2":    {"N": 2},
    "AR":    {"Ar": 1},
    "HE":    {"He": 1},
    "CO":    {"C": 1, "O": 1},
    "CO2":   {"C": 1, "O": 2},
    "CH4":   {"C": 1, "H": 4},
    "CH3":   {"C": 1, "H": 3},
    "CH2O":  {"C": 1, "H": 2, "O": 1},
    "HCO":   {"C": 1, "H": 1, "O": 1},
    "CH2":   {"C": 1, "H": 2},
    "CH2(S)":{"C": 1, "H": 2},
    "CH":    {"C": 1, "H": 1},
    "C2H6":  {"C": 2, "H": 6},
    "C2H5":  {"C": 2, "H": 5},
    "C2H4":  {"C": 2, "H": 4},
    "C2H3":  {"C": 2, "H": 3},
    "C2H2":  {"C": 2, "H": 2},
    "C2H":   {"C": 2, "H": 1},
}


# ─────────────────────────────────────────────────────────────────────────────
# Kinetics parser — convert RMG string repr → Cantera YAML fields
# ─────────────────────────────────────────────────────────────────────────────

def _extract_numeric(raw: str) -> float | None:
    """
    Extract the first numeric value from a raw string that may be:
      - a plain float:          '38700'
      - a list [value, unit]:   "[20000000000000.0, 'cm^3/(mol*s)']"
      - scientific notation:    '3.87e+04'
    Returns float or None.
    """
    if raw is None:
        return None
    raw = str(raw).strip()
    # Try direct parse first
    try:
        return float(raw)
    except ValueError:
        pass
    # Extract first number from list format like "[1.23e+13, 'cm^3/(mol*s)']"
    m = re.search(r'[\[\(]?\s*([\d.eE+\-]+)', raw)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def parse_troe_params(kinetics_str: str) -> dict | None:
    """
    Parse a full Troe kinetics string into Cantera falloff parameters.

    Input format (from RMG extraction):
      Troe(arrheniusHigh={'kwargs': {'A': [5.1e12,'cm^3/(mol*s)'], 'n': 0.44, 'Ea': [0,'cal/mol']}},
           arrheniusLow={'kwargs':  {'A': [6.3e19,'cm^6/(mol^2*s)'], 'n': -1.4, 'Ea': [0,'cal/mol']}},
           alpha=0.5, T3=[1e-30,'K'], T1=[1e+30,'K'], T2=[5182,'K'],
           efficiencies={'O': 11.89, '[H][H]': 2.0, '[Ar]': 0.4, ...})

    Returns:
      {
        'high_P': {'A': ..., 'b': ..., 'Ea': ...},
        'low_P':  {'A': ..., 'b': ..., 'Ea': ...},
        'alpha':   float,
        'T3':      float,
        'T1':      float,
        'T2':      float | None,
        'efficiencies': {smiles: float, ...}  — SMILES keys, convert with species_tool
      }
    Returns None if parsing fails.
    """
    if not isinstance(kinetics_str, str):
        return None
    ks = kinetics_str.strip()
    if not ks.startswith(("Troe(", "Lindemann(")):
        return None

    high_P = _extract_nested_arrhenius(ks, "arrheniusHigh")
    low_P  = _extract_nested_arrhenius(ks, "arrheniusLow")

    if high_P is None and low_P is None:
        return None

    # ── Troe shape parameters ──────────────────────────────────────────────
    alpha = None
    T3 = T1 = T2 = None

    alpha_m = re.search(r'\balpha\s*=\s*(-?[\d.eE+\-]+)', ks)
    if alpha_m:
        try:
            alpha = float(alpha_m.group(1))
        except ValueError:
            pass

    def _extract_troe_T(name: str) -> float | None:
        # Matches: T3=91 or T3=[91,'K'] or T3=(91,'K')
        m = re.search(rf'\b{name}\s*=\s*[\[\(]?\s*(-?[\d.eE+\-]+)', ks)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
        return None

    T3 = _extract_troe_T("T3")
    T1 = _extract_troe_T("T1")
    T2 = _extract_troe_T("T2")

    # ── Efficiencies dict ─────────────────────────────────────────────────
    efficiencies: dict = {}
    eff_m = re.search(r'efficiencies\s*=\s*(\{[^}]+\})', ks)
    if eff_m:
        try:
            import ast
            efficiencies = ast.literal_eval(eff_m.group(1))
        except Exception:
            pass

    return {
        "high_P":       high_P,
        "low_P":        low_P,
        "alpha":        alpha,
        "T3":           T3,
        "T1":           T1,
        "T2":           T2,
        "efficiencies": efficiencies,
    }


def parse_arrhenius_string(kinetics_str, kinetics_type: str = "") -> dict | None:
    """
    Parse RMG kinetics into {A, b, Ea} suitable for Cantera rate-constant field.

    Handles:
      Arrhenius:  Arrhenius(A=[2e13, 'cm^3/(mol*s)'], n=0, Ea=[0, 'cal/mol'], ...)
      ThirdBody:  ThirdBody(arrheniusLow={...'A': [1.78e18,...], 'n': -1, 'Ea': [0,...]}, ...)
      Troe:       Troe(arrheniusHigh={...'A': [5.1e12,...], 'n': 0.44, 'Ea': [0,...]}, ...)
      Lindemann:  Lindemann(arrheniusHigh={...}, arrheniusLow={...}, ...)
      Dict:       {'A': (2e13, 'cm^3/(mol*s)'), 'n': 0, 'Ea': 0}
      MultiArrheniusreturns first term
    """
    # ── Dict ──────────────────────────────────────────────────────────────────
    if isinstance(kinetics_str, dict):
        A  = _extract_numeric(str(kinetics_str.get('A', kinetics_str.get('a', None))))
        n  = _extract_numeric(str(kinetics_str.get('n', kinetics_str.get('b', 0))))
        Ea = _extract_numeric(str(kinetics_str.get('Ea', kinetics_str.get('ea', 0))))
        if A is not None:
            return {"A": A, "b": n or 0.0, "Ea": Ea or 0.0}
        return None

    if not isinstance(kinetics_str, str):
        return None

    ks = kinetics_str.strip()

    # ── ThirdBody / Troe / Lindemann: extract from nested arrheniusHigh or arrheniusLow ──
    # For Cantera simple Arrhenius representation we use:
    #   Troe/Lindemann → arrheniusHigh (high-pressure limit)
    #   ThirdBody      → arrheniusLow  (the low-pressure rate IS the rate)
    #   (Cantera handles the falloff internally; we just need a valid A/n/Ea)
    if ks.startswith(("Troe(", "Lindemann(")):
        # Extract arrheniusHigh first, fall back to arrheniusLow
        for key in ("arrheniusHigh", "arrheniusLow"):
            block = _extract_nested_arrhenius(ks, key)
            if block:
                return block
        return None

    if ks.startswith("ThirdBody("):
        block = _extract_nested_arrhenius(ks, "arrheniusLow")
        if block:
            return block
        return None

    if ks.startswith("MultiArrhenius("):
        # Take the first Arrhenius term
        # Find first 'A': [...] pattern inside
        return _extract_first_multiarrhenius(ks)

    if ks.startswith("MultiPDepArrhenius("):
        return None  # too complex for POC

    # ── Standard Arrhenius ─────────────────────────────────────────────────────
    try:
        A_match = re.search(
            r'\bA\s*=\s*(?:[\[\(]\s*([\d.eE+\-]+)|([  .\d.eE+\-]+))',
            ks
        )
        # Simpler fallback — find A= anywhere
        if not A_match:
            A_match = re.search(r'[\'"]A[\'"]\s*:\s*[\[\(]?\s*([\d.eE+\-]+)', ks)
            if A_match:
                A = float(A_match.group(1))
            else:
                logger.debug(f"No A= found in: {ks[:120]}")
                return None
        else:
            A = float(A_match.group(1) or A_match.group(2))

        n_match  = re.search(r'\bn\s*=\s*([\d.eE+\-]+)', ks)
        Ea_match = re.search(
            r'\bEa\s*=\s*(?:[\[\(]\s*([\d.eE+\-]+)|([\d.eE+\-]+))', ks
        )
        n  = float(n_match.group(1)) if n_match else 0.0
        Ea = float((Ea_match.group(1) or Ea_match.group(2))) if Ea_match else 0.0

        return {"A": A, "b": n, "Ea": Ea}

    except (AttributeError, ValueError, TypeError) as e:
        logger.debug(f"Arrhenius parse failed ({e}): {ks[:120]!r}")
        return None


def _extract_nested_arrhenius(kinetics_str: str, key: str) -> dict | None:
    """
    Extract A/n/Ea from a nested dict-like block such as:
      arrheniusHigh={'_type': 'Call', 'func': 'Arrhenius', 'args': [],
                     'kwargs': {'A': [5.1e12, 'cm^3/(mol*s)'], 'n': 0.44, 'Ea': [0,'cal/mol']}}
    """
    # Find the block starting at key=
    idx = kinetics_str.find(key + "=")
    if idx == -1:
        idx = kinetics_str.find(f"'{key}'")
        if idx == -1:
            return None

    # Extract from 'kwargs' sub-dict which contains A, n, Ea
    # Look for 'A': [value, ...] after the key position
    sub = kinetics_str[idx:]

    # Find A value — format: 'A': [1.23e+13, 'unit'] or 'A': 1.23e+13
    A_match = re.search(r"['\"]A['\"]\s*:\s*(?:[\[\(]\s*([\d.eE+\-]+)|([\d.eE+\-]+))", sub)
    n_match  = re.search(r"['\"]n['\"]\s*:\s*([\d.eE+\-]+)", sub)
    Ea_match = re.search(r"['\"]Ea['\"]\s*:\s*(?:[\[\(]\s*(-?[\d.eE+\-]+)|(-?[\d.eE+\-]+))", sub)

    if not A_match:
        return None

    try:
        A  = float(A_match.group(1) or A_match.group(2))
        n  = float(n_match.group(1)) if n_match else 0.0
        Ea = float((Ea_match.group(1) or Ea_match.group(2))) if Ea_match else 0.0
        return {"A": A, "b": n, "Ea": Ea}
    except (ValueError, TypeError):
        return None


def _extract_first_multiarrhenius(kinetics_str: str) -> dict | None:
    """Extract A/n/Ea from the first term of a MultiArrhenius string."""
    # Find first 'A': pattern
    A_match = re.search(r"['\"]A['\"]\s*:\s*(?:[\[\(]\s*([\d.eE+\-]+)|([\d.eE+\-]+))", kinetics_str)
    if not A_match:
        return None
    sub_start = A_match.start()
    sub = kinetics_str[max(0, sub_start - 50): sub_start + 200]
    n_match  = re.search(r"['\"]n['\"]\s*:\s*([\d.eE+\-]+)", sub)
    Ea_match = re.search(r"['\"]Ea['\"]\s*:\s*(?:[\[\(]\s*(-?[\d.eE+\-]+)|(-?[\d.eE+\-]+))", sub)
    try:
        A  = float(A_match.group(1) or A_match.group(2))
        n  = float(n_match.group(1)) if n_match else 0.0
        Ea = float((Ea_match.group(1) or Ea_match.group(2))) if Ea_match else 0.0
        return {"A": A, "b": n, "Ea": Ea}
    except (ValueError, TypeError):
        return None


def _canonical_reaction_key(label: str) -> str:
    """
    Normalize a reaction label to a canonical form for de-duplication.
    Handles species-order differences: 'O + HO2 <=> OH + O2' == 'HO2 + O <=> OH + O2'.
    Strips third-body notation, sorts species on each side, lowercases.
    """
    label = re.sub(r'\s*\(\+?M\)\s*', '', label)   # remove (+M) and (M)
    label = re.sub(r'\s*\+\s*M\b', '', label)       # remove + M
    label = label.strip()

    # Split on <=> or => or =
    for sep in ['<=>', '=>', '=']:
        if sep in label:
            parts = label.split(sep, 1)
            lhs = tuple(sorted(s.strip() for s in parts[0].split('+')))
            rhs = tuple(sorted(s.strip() for s in parts[1].split('+')))
            # Canonical: always put lexicographically smaller side first
            if lhs > rhs:
                lhs, rhs = rhs, lhs
            return f"{'+'.join(lhs)}<=>{'+'.join(rhs)}".lower()

    return label.lower()


def _is_third_body(label: str, kinetics_type: str = "") -> bool:
    """
    Determine if a reaction is a third-body reaction.

    Uses kinetics_type as the authoritative signal (from RMG DB field),
    with label-string detection as a fallback for reactions not in the DB.
    """
    # Primary: kinetics type from DB — this is authoritative
    if kinetics_type in ("ThirdBody", "Troe", "Lindemann", "PDepArrhenius", "MultiPDepArrhenius"):
        return True
    # Secondary: label string patterns (for injected reactions or non-DB sources)
    label_up = label.upper()
    return "(+M)" in label_up or "+ M " in label_up or label_up.strip().endswith("+M")


def _canonical_tb_equation(label: str, reactants: list, products: list) -> str:
    """
    Build the Cantera equation string for a three-body reaction.

    In Cantera YAML, reactions with `type: three-body` MUST include '+ M'
    on BOTH sides of the equation symmetrically:
      H + O2 + M <=> HO2 + M
      2 H + M <=> H2 + M
      H + OH + M <=> H2O + M

    The DB stores labels and reactants/products WITHOUT M (M is implicit in
    kinetics_type=ThirdBody/Troe). This function reconstructs the proper form.
    """
    from collections import Counter

    def format_side(species_list: list) -> str:
        # Filter out any M placeholders if they snuck in
        sp = [s for s in species_list if s.upper() not in ("M", "(M)", "(+M)")]
        counts = Counter(s.upper() for s in sp)
        parts = []
        for name, n in counts.items():
            parts.append(f"{n} {name}" if n > 1 else name)
        return " + ".join(parts)

    if reactants and products:
        lhs = format_side(reactants) + " + M"
        rhs = format_side(products)  + " + M"
        if lhs and rhs:
            return f"{lhs} <=> {rhs}"

    # Fallback: normalise existing label and ensure symmetric + M
    eq = label.strip()
    # Strip any existing M notation first, then re-add symmetrically
    eq = re.sub(r'\s*\(\+?M\)\s*', '', eq)
    eq = re.sub(r'\s*\+\s*M\b', '', eq)
    eq = re.sub(r'\s+', ' ', eq).strip()
    # Now add + M to both sides
    if '<=>' in eq:
        lhs, rhs = eq.split('<=>', 1)
        return f"{lhs.strip()} + M <=> {rhs.strip()} + M"
    return eq


def _canonical_falloff_equation(label: str, reactants: list, products: list) -> str:
    """
    Build the Cantera equation string for a falloff (Troe/Lindemann) reaction.
    Falloff reactions use (+M) on both sides: H + O2 (+M) <=> HO2 (+M)
    """
    from collections import Counter

    def format_side(species_list: list) -> str:
        sp = [s for s in species_list if s.upper() not in ("M", "(M)", "(+M)")]
        counts = Counter(s.upper() for s in sp)
        parts = []
        for name, n in counts.items():
            parts.append(f"{n} {name}" if n > 1 else name)
        return " + ".join(parts)

    if reactants and products:
        lhs = format_side(reactants) + " (+M)"
        rhs = format_side(products)  + " (+M)"
        if lhs and rhs:
            return f"{lhs} <=> {rhs}"

    # Fallback: normalize existing label and ensure symmetric (+M)
    eq = label.strip()
    eq = re.sub(r'\s*\(\+?M\)\s*', '', eq)
    eq = re.sub(r'\s*\+\s*M\b', '', eq)
    eq = re.sub(r'\s+', ' ', eq).strip()
    if '<=>' in eq:
        lhs, rhs = eq.split('<=>', 1)
        return f"{lhs.strip()} (+M) <=> {rhs.strip()} (+M)"
    return eq


def _is_falloff(kinetics_str: str) -> bool:
    return "Troe" in kinetics_str or "Lindemann" in kinetics_str


# ─────────────────────────────────────────────────────────────────────────────
# Main builder
# ─────────────────────────────────────────────────────────────────────────────

def build_mechanism_yaml(
    reactions: list[dict],
    mechanism_name: str = "Agent-Generated H2/O2 Mechanism",
    description: str = "",
    extra_species: list[str] | None = None,
) -> tuple[str, dict]:
    """
    Build a Cantera YAML mechanism from a list of reaction dicts.

    Returns:
        (yaml_string, build_report)
    """
    # Collect all species mentioned in reactions
    all_species = set(extra_species or [])
    valid_reactions = []
    skipped = []

    for rxn in reactions:
        label = rxn.get("label", "")
        kin   = rxn.get("kinetics", "")

        # Handle supported kinetics types
        supported = ("Arrhenius", "ThirdBody", "Troe", "Lindemann", "MultiArrhenius", None, "")
        if rxn.get("kinetics_type") not in supported:
            skipped.append({"label": label, "reason": f"kinetics type {rxn.get('kinetics_type')} not yet supported"})
            continue

        params = parse_arrhenius_string(kin, rxn.get("kinetics_type", ""))
        if params is None:
            skipped.append({"label": label, "reason": "could not parse Arrhenius params"})
            continue

        # Collect species
        for sp in rxn.get("reactants", []) + rxn.get("products", []):
            all_species.add(sp.upper())

        valid_reactions.append({
            "label":         label,
            "reactants":     rxn.get("reactants", []),
            "products":      rxn.get("products",  []),
            "params":        params,
            "kinetics":      kin,
            "duplicate":     rxn.get("duplicate", False),
            "kinetics_type": rxn.get("kinetics_type", "Arrhenius"),
            "third_body":    _is_third_body(label, rxn.get("kinetics_type", "")),
        })

    # Filter species to those we have thermo data for
    known_species  = [sp for sp in all_species if sp in NASA7_DATA]
    unknown_species = [sp for sp in all_species if sp not in NASA7_DATA]

    if unknown_species:
        logger.warning(f"No thermo data for: {unknown_species} — excluded from mechanism")

    # De-duplicate reactions by CANONICAL form (catches O+HO2 == HO2+O)
    seen_keys = set()
    unique_reactions = []
    for r in valid_reactions:
        key = _canonical_reaction_key(r["label"])
        if key not in seen_keys:
            seen_keys.add(key)
            unique_reactions.append(r)
        else:
            skipped.append({"label": r["label"], "reason": "canonical duplicate"})

    # Drop reactions that reference species with no thermo data — Cantera will
    # reject them anyway, and keeping them causes species-undeclared errors.
    known_sp_set = set(known_species)
    filtered_reactions = []
    for r in unique_reactions:
        rxn_species = {sp.upper() for sp in r.get("reactants", []) + r.get("products", [])}
        missing = rxn_species - known_sp_set
        if missing:
            skipped.append({"label": r["label"], "reason": f"species without thermo data: {missing}"})
        else:
            filtered_reactions.append(r)
    unique_reactions = filtered_reactions

    yaml_str = _render_yaml(unique_reactions, known_species, mechanism_name, description)

    report = {
        "mechanism_name": mechanism_name,
        "n_species":      len(known_species),
        "n_reactions":    len(unique_reactions),
        "species":        known_species,
        "skipped_reactions": skipped,
        "unknown_species_excluded": unknown_species,
        "valid": len(unique_reactions) > 0 and len(known_species) >= 3,
    }

    return yaml_str, report


# ─────────────────────────────────────────────────────────────────────────────
# YAML renderer
# ─────────────────────────────────────────────────────────────────────────────

def _render_yaml(reactions: list[dict], species: list[str], name: str, desc: str) -> str:
    lines = []

    # Header
    lines += [
        f"description: |",
        f"  {name}",
        f"  {desc}",
        f"  Generated by CombustionAgent POC",
        f"",
        f"units: {{length: cm, time: s, quantity: mol, activation-energy: cal/mol}}",
        f"",
    ]

    # Phases
    sp_list = ", ".join(species)
    lines += [
        f"phases:",
        f"- name: gas",
        f"  thermo: ideal-gas",
        f"  elements: [H, O, N, Ar]",
        f"  species: [{sp_list}]",
        f"  kinetics: gas",
        f"  explicit-third-body-duplicates: mark-duplicate",
        f"  state: {{T: 300.0, P: 1 atm}}",
        f"",
    ]

    # Species thermodynamics
    lines += ["species:"]
    for sp in species:
        d = NASA7_DATA[sp]
        hi = d["hi"]
        lo = d["lo"]
        mw = MOLECULAR_WEIGHTS.get(sp, 1.0)
        lines += [
            f"- name: {sp}",
            f"  composition: {_composition(sp)}",
            f"  thermo:",
            f"    model: NASA7",
            f"    temperature-ranges: [{d['Tmin']}, {d['Tmid']}, {d['Tmax']}]",
            f"    data:",
            f"    - [{', '.join(f'{v:.8e}' for v in hi)}]",
            f"    - [{', '.join(f'{v:.8e}' for v in lo)}]",
            f"",
        ]

    # Reactions
    # Track equations already emitted to catch same-equation duplicates (Cantera needs duplicate: true)
    seen_equations = {}
    lines += ["reactions:"]
    for rxn in reactions:
        p        = rxn["params"]
        label    = rxn["label"]
        equation = _normalize_equation(label)

        is_dup = rxn.get("duplicate", False)
        # If we've seen this exact equation before, mark both as duplicate
        if equation in seen_equations:
            is_dup = True
            # Retroactively mark the earlier entry
            seen_equations[equation]["dup"] = True
        else:
            entry = {"dup": is_dup}
            seen_equations[equation] = entry

        lines += [
            f"- equation: {equation}",
            f"  rate-constant: {{A: {p['A']:.6e}, b: {p['b']:.3f}, Ea: {p['Ea']:.2f}}}",
        ]
        if is_dup:
            lines.append(f"  duplicate: true")
        lines.append("")

    # Retroactive duplicate flagging pass — rewrite lines for any back-marked entries
    # (simpler: just rebuild with duplicate awareness)
    final_lines = _rerender_reactions_with_dup_flags(reactions, species)
    # Replace reactions section
    header_end = next(i for i, l in enumerate(lines) if l == "reactions:")
    lines = lines[:header_end] + final_lines

    return "\n".join(lines)


def _rerender_reactions_with_dup_flags(reactions: list[dict], species: list[str] | None = None) -> list[str]:
    """
    Two-pass rendering: determine which equations appear more than once,
    then emit all with duplicate: true where needed.

    Reaction types handled:
      - Troe/Lindemann → type: falloff  with high-P/low-P rate constants + Troe params
      - ThirdBody      → type: three-body with rate-constant and efficiencies
      - Arrhenius/Multi → simple rate-constant block

    Third-body efficiencies are filtered to species actually declared in the
    mechanism — Cantera rejects efficiency entries for undeclared species.
    SMILES efficiency keys (from Troe blocks) are converted to species names.
    """
    from collections import Counter
    try:
        from tools.species_tool import SMILES_TO_NAME
    except ImportError:
        try:
            from species_tool import SMILES_TO_NAME
        except ImportError:
            SMILES_TO_NAME = {}

    # Standard three-body efficiency defaults (applied when DB has none)
    DEFAULT_EFFICIENCIES = {"H2": 2.4, "H2O": 15.4, "AR": 0.67}
    declared = set(species) if species else set()

    _FALLOFF_TYPES = {"Troe", "Lindemann"}
    _TB_TYPE       = "ThirdBody"

    # Build canonical equations first (needed for dedup counting)
    def get_equation(rxn: dict) -> str:
        kt = rxn.get("kinetics_type", "")
        if kt in _FALLOFF_TYPES:
            return _canonical_falloff_equation(
                rxn["label"], rxn.get("reactants", []), rxn.get("products", [])
            )
        is_tb = rxn.get("third_body", False) or rxn.get("_injected", False)
        if is_tb:
            return _canonical_tb_equation(
                rxn["label"], rxn.get("reactants", []), rxn.get("products", [])
            )
        return _normalize_equation(rxn["label"])

    equation_counts = Counter(get_equation(r) for r in reactions)

    def _filter_efficiencies(raw_eff: dict) -> dict:
        """Convert SMILES keys → species names and filter to declared species."""
        result = {}
        for key, val in raw_eff.items():
            # Try SMILES → name lookup first
            name = SMILES_TO_NAME.get(key, key)
            name_up = name.upper()
            if name_up in declared:
                result[name_up] = val
        return result

    lines = ["reactions:"]
    for rxn in reactions:
        p        = rxn["params"]
        kt       = rxn.get("kinetics_type", "Arrhenius")
        equation = get_equation(rxn)
        # Only use count-based duplicate detection — never trust the DB's
        # duplicate flag blindly, because that flag assumes both copies of the
        # pair are present, which is not guaranteed in a growing mechanism.
        is_dup = equation_counts[equation] > 1

        if kt in _FALLOFF_TYPES:
            # ── Troe / Lindemann falloff ──────────────────────────────────
            troe = parse_troe_params(rxn.get("kinetics", ""))
            if troe and troe.get("high_P") and troe.get("low_P"):
                hp = troe["high_P"]
                lp = troe["low_P"]
                lines += [
                    f"- equation: {equation}",
                    f"  type: falloff",
                    f"  high-P-rate-constant: {{A: {hp['A']:.6e}, b: {hp['b']:.3f}, Ea: {hp['Ea']:.2f}}}",
                    f"  low-P-rate-constant:  {{A: {lp['A']:.6e}, b: {lp['b']:.3f}, Ea: {lp['Ea']:.2f}}}",
                ]
                # Troe shape params (omit if Lindemann or missing)
                if troe.get("alpha") is not None and troe.get("T3") is not None:
                    troe_fields = f"A: {troe['alpha']}, T3: {troe['T3']:.6e}, T1: {troe['T1']:.6e}"
                    if troe.get("T2") is not None:
                        troe_fields += f", T2: {troe['T2']:.6e}"
                    lines.append(f"  Troe: {{{troe_fields}}}")
                # Efficiencies from the parsed kinetics string
                raw_eff = troe.get("efficiencies", {})
                eff = _filter_efficiencies(raw_eff) if raw_eff else {k: v for k, v in DEFAULT_EFFICIENCIES.items() if k in declared}
                if not eff:
                    eff = {k: v for k, v in DEFAULT_EFFICIENCIES.items() if k in declared}
                if eff:
                    eff_str = ", ".join(f"{k}: {v}" for k, v in eff.items())
                    lines.append(f"  efficiencies: {{{eff_str}}}")
            else:
                # parse_troe_params failed — fall back to simple rate-constant
                lines += [
                    f"- equation: {equation}",
                    f"  rate-constant: {{A: {p['A']:.6e}, b: {p['b']:.3f}, Ea: {p['Ea']:.2f}}}",
                ]

        elif kt == _TB_TYPE or rxn.get("third_body", False) or rxn.get("_injected", False):
            # ── ThirdBody three-body ──────────────────────────────────────
            eff = {k: v for k, v in DEFAULT_EFFICIENCIES.items() if k in declared}
            lines += [
                f"- equation: {equation}",
                f"  type: three-body",
                f"  rate-constant: {{A: {p['A']:.6e}, b: {p['b']:.3f}, Ea: {p['Ea']:.2f}}}",
            ]
            if eff:
                eff_str = ", ".join(f"{k}: {v}" for k, v in eff.items())
                lines.append(f"  efficiencies: {{{eff_str}}}")

        else:
            # ── Simple Arrhenius ──────────────────────────────────────────
            lines += [
                f"- equation: {equation}",
                f"  rate-constant: {{A: {p['A']:.6e}, b: {p['b']:.3f}, Ea: {p['Ea']:.2f}}}",
            ]

        if is_dup:
            lines.append(f"  duplicate: true")
        lines.append("")
    return lines




def _composition(sp: str) -> str:
    """
    Return element composition YAML fragment for a species.
    Uses _COMPOSITIONS lookup first; falls back to parsing the formula
    from the species name if not found (handles common Hill-notation names).
    """
    comp = _COMPOSITIONS.get(sp.upper())
    if comp:
        parts = ", ".join(f"{el}: {n}" for el, n in comp.items())
        return "{" + parts + "}"

    # Fallback: parse Hill-notation formula from species name
    # Handles strings like C3H8, C12H26, etc.
    counts: dict[str, int] = {}
    for el, n_str in re.findall(r'([A-Z][a-z]?)(\d*)', sp):
        if el:
            counts[el] = counts.get(el, 0) + (int(n_str) if n_str else 1)
    if counts:
        parts = ", ".join(f"{el}: {n}" for el, n in counts.items())
        return "{" + parts + "}"

    logger.warning(f"Unknown composition for species: {sp!r} — defaulting to {{C: 1}}")
    return "{C: 1}"


def _normalize_equation(label: str) -> str:
    """
    Normalize reaction label to Cantera equation format.
    RMG uses <=> for reversible, Cantera accepts this.
    Strip third-body (M) notation for simple Arrhenius reactions.
    """
    eq = label.strip()
    # Replace common formatting issues
    eq = eq.replace("(+M)", "").replace("+ M ", "").replace("+M", "")
    eq = re.sub(r'\s+', ' ', eq).strip()
    return eq


# ─────────────────────────────────────────────────────────────────────────────
# Tool schema + dispatcher
# ─────────────────────────────────────────────────────────────────────────────

TOOL_SCHEMAS = [
    {
        "name": "build_mechanism",
        "description": (
            "Assemble a Cantera YAML mechanism from a list of reactions retrieved from the DB. "
            "Returns the YAML string and a build report (species count, skipped reactions, etc.). "
            "Call this after get_reactions_for_fuel to convert the raw reactions into a runnable mechanism."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reactions": {
                    "type": "array",
                    "description": "List of reaction dicts from DB retrieval tool",
                    "items": {"type": "object"},
                },
                "mechanism_name": {"type": "string", "description": "Name for the mechanism"},
                "description":    {"type": "string", "description": "Optional description"},
                "extra_species":  {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Additional species to include (e.g. bath gases: N2, AR)",
                },
            },
            "required": ["reactions"],
        },
    },
]


def dispatch(tool_name: str, args: dict) -> str:
    try:
        if tool_name == "build_mechanism":
            yaml_str, report = build_mechanism_yaml(
                args["reactions"],
                mechanism_name=args.get("mechanism_name", "Agent Mechanism"),
                description=args.get("description", ""),
                extra_species=args.get("extra_species", ["N2", "AR"]),
            )
            return json.dumps({
                "mechanism_yaml": yaml_str,
                "report": report,
            })
        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
    except Exception as e:
        logger.exception(f"Mechanism builder error")
        return json.dumps({"error": str(e)})