"""
OrchestratorState — single source of truth for the multi-agent loop.

All subagents receive a READ-ONLY view of this state.
Only the orchestrator mutates it.
"""

from dataclasses import dataclass, field
from typing import Any

import config
from mechanism_builder import NASA7_DATA as _NASA7_DATA

# Set of species with built-in thermo data — exposed to selector so it can
# prefer reactions where all species are known. Updated automatically when
# NASA7_DATA is extended.
KNOWN_THERMO_SPECIES: frozenset[str] = frozenset(_NASA7_DATA.keys())

_TB_TYPES = {"ThirdBody", "Troe", "Lindemann"}

def _annotate_label(r: dict) -> str:
    """
    Return the reaction label annotated with its kinetics class.
    ThirdBody/Troe/Lindemann reactions are annotated with their rendered
    Cantera equation (with +M) so the LLM agents understand these ARE
    third-body reactions even though the DB label lacks '+M'.
    """
    kt = r.get("kinetics_type", "Arrhenius")
    if kt in _TB_TYPES:
        return f"{r['label']} [three-body → rendered with +M in YAML]"
    return f"{r['label']} [{kt}]"


@dataclass
class AdditionLogEntry:
    """Records why each reaction was added — the audit trail."""
    iteration:    int
    added_labels: list[str]
    reason:       str
    score_before: float
    score_after:  float = 0.0   # filled in after validation


@dataclass
class ValidationRecord:
    iteration:        int
    score:            float
    n_successful:     int
    n_conditions:     int
    idt_results:      list[dict]
    failure_mode:     str = ""   # filled by diagnostician


@dataclass
class DiagnosticianReport:
    """Output contract for the Diagnostician agent."""
    failure_mode:      str          # human-readable: "missing chain termination"
    missing_pathways:  list[str]    # e.g. ["H + O2 + M -> HO2 + M", "H + OH + M -> H2O + M"]
    target_species:    list[str]    # species the selector should prioritise
    severity:          str          # "fatal" | "partial" | "ok"
    raw_cantera_error: str = ""


@dataclass
class SelectorOutput:
    """Output contract for the Chemistry Selector agent."""
    selected:             list[dict]   # list of reaction dicts from candidate_pool
    reasoning:            str          # chain-of-thought explanation
    next_target_species:  list[str]    # species to prioritise in next DB search


@dataclass
class OrchestratorState:
    # ── Task ──────────────────────────────────────────────────────────────────
    fuel:              str = "H2"
    target_conditions: list[dict] = field(default_factory=lambda: [
        {"T": 800,  "P": 101325, "phi": 1.0, "label": "T800_P1atm"},
        {"T": 1000, "P": 101325, "phi": 1.0, "label": "T1000_P1atm"},
        {"T": 1200, "P": 101325, "phi": 1.0, "label": "T1200_P1atm"},
        {"T": 1000, "P": 506625, "phi": 1.0, "label": "T1000_P5atm"},
    ])
    acceptance_threshold: float = 0.75

    # ── Current mechanism ─────────────────────────────────────────────────────
    current_mechanism_yaml: str = ""
    current_reactions:      list[dict] = field(default_factory=list)
    current_species:        list[str]  = field(default_factory=list)

    # ── Search state ──────────────────────────────────────────────────────────
    candidate_pool:         list[dict] = field(default_factory=list)
    already_tried_labels:   set        = field(default_factory=set)
    # Populated from FUEL_CONFIGS in __post_init__ — not hardcoded here
    target_species:         list[str]  = field(default_factory=list)
    library_preference:     list[str]  = field(default_factory=list)
    bath_gases:             list[str]  = field(default_factory=list)
    oxidizer:               str        = "O2"

    # ── History ───────────────────────────────────────────────────────────────
    iteration:            int = 0
    best_score:           float = 0.0
    best_mechanism_yaml:  str = ""
    best_reactions_snapshot: list[dict] = field(default_factory=list)
    validation_history:   list[ValidationRecord]    = field(default_factory=list)
    addition_log:         list[AdditionLogEntry]    = field(default_factory=list)
    diagnostician_report: DiagnosticianReport | None = None
    last_cantera_result:  dict                       = field(default_factory=dict)
    flux_analysis:        dict                       = field(default_factory=dict)

    # ── Run metadata ──────────────────────────────────────────────────────────
    run_id:    str = ""
    max_iters: int = 8

    def __post_init__(self):
        """Initialize fuel-specific fields from FUEL_CONFIGS if not explicitly set."""
        fuel_cfg = config.get_fuel_config(self.fuel)
        if not self.target_species:
            self.target_species = list(fuel_cfg["initial_species"])
        if not self.library_preference:
            self.library_preference = list(fuel_cfg["library_preference"])
        if not self.bath_gases:
            self.bath_gases = list(fuel_cfg["bath_gases"])
        if self.oxidizer == "O2":
            self.oxidizer = fuel_cfg.get("oxidizer", "O2")

    # ─────────────────────────────────────────────────────────────────────────
    # Context builders — each subagent gets exactly what it needs
    # ─────────────────────────────────────────────────────────────────────────

    def context_for_db_search(self) -> dict:
        """Minimal context for the DB search agent."""
        return {
            "fuel": self.fuel,
            "target_species": self.target_species,
            "already_tried_labels": list(self.already_tried_labels),
            "current_reaction_labels": [r["label"] for r in self.current_reactions],
            "library_preference": self.library_preference,
            "iteration": self.iteration,
        }

    def context_for_selector(self) -> dict:
        """Full chemistry context for the selector — this is the richest payload."""
        return {
            # Fuel identity — allows fuel-agnostic reasoning
            "fuel": self.fuel,
            "oxidizer": self.oxidizer,
            # Species with available thermo data — selector should prefer reactions
            # where all species are in this set; others will be skipped at build time
            "known_thermo_species": sorted(KNOWN_THERMO_SPECIES),
            # What's already in the mechanism (annotated with kinetics type)
            "current_reaction_labels": [
                _annotate_label(r) for r in self.current_reactions
            ],
            "current_species": self.current_species,
            "n_current_reactions": len(self.current_reactions),

            # What the diagnostician said is missing
            "diagnostician_report": {
                "failure_mode":     self.diagnostician_report.failure_mode     if self.diagnostician_report else "unknown — first iteration",
                "missing_pathways": self.diagnostician_report.missing_pathways if self.diagnostician_report else [],
                "target_species":   self.diagnostician_report.target_species   if self.diagnostician_report else self.target_species,
                "severity":         self.diagnostician_report.severity         if self.diagnostician_report else "fatal",
            },

            # Score history so selector knows the trend
            "validation_history": [
                {"iteration": v.iteration, "score": v.score, "failure_mode": v.failure_mode}
                for v in self.validation_history
            ],

            # Full audit trail — selector must NOT re-add these
            "addition_log": [
                {"iteration": e.iteration, "added": e.added_labels, "reason": e.reason}
                for e in self.addition_log
            ],

            # The candidate pool — selector MUST only pick from these
            "candidate_pool_labels": [r["label"] for r in self.candidate_pool],
            "candidate_pool_count":  len(self.candidate_pool),

            # How many to select this round
            "n_to_select": self._n_to_select(),
            "iteration": self.iteration,
        }

    def context_for_diagnostician(self) -> dict:
        """Cantera result + mechanism context for diagnostician."""
        # Summarise IDT results — strip large arrays
        idt_summary = []
        for r in self.last_cantera_result.get("idt_results", []):
            idt_summary.append({
                "condition": r.get("condition_label"),
                "success":   r.get("success"),
                "idt_ms":    r.get("ignition_delay_ms"),
                "error":     r.get("error", ""),
                "peak_T":    r.get("peak_T"),
            })

        return {
            "fuel": self.fuel,
            "oxidizer": self.oxidizer,
            "iteration": self.iteration,
            "current_reaction_labels": [
                _annotate_label(r) for r in self.current_reactions
            ],
            "current_species": self.current_species,
            "flux_analysis": self.flux_analysis,
            "validation_score": self.last_cantera_result.get("validation_score", 0.0),
            "idt_results": idt_summary,
            "validation_history": [
                {"iteration": v.iteration, "score": v.score}
                for v in self.validation_history
            ],
            "last_addition": (
                {"added": self.addition_log[-1].added_labels,
                 "reason": self.addition_log[-1].reason}
                if self.addition_log else None
            ),
        }

    def _n_to_select(self) -> int:
        """How many reactions to add this iteration.
        Start conservative (1), increase if stuck.
        """
        if self.iteration == 0:
            return 3   # seed with a few core reactions
        recent_scores = [v.score for v in self.validation_history[-3:]]
        if len(recent_scores) >= 2 and max(recent_scores) - min(recent_scores) < 0.05:
            return 3   # stuck — try more
        return 1       # making progress — stay surgical

    # ─────────────────────────────────────────────────────────────────────────
    # State mutation helpers (only orchestrator calls these)
    # ─────────────────────────────────────────────────────────────────────────

    def add_reactions(self, new_reactions: list[dict], reason: str):
        """Append reactions and update already_tried set."""
        score_before = self.best_score
        for r in new_reactions:
            label = r["label"]
            if label not in {x["label"] for x in self.current_reactions}:
                self.current_reactions.append(r)
                self.already_tried_labels.add(label)
                # Track all species
                for sp in r.get("reactants", []) + r.get("products", []):
                    sp_up = sp.upper()
                    if sp_up not in self.current_species:
                        self.current_species.append(sp_up)

        self.addition_log.append(AdditionLogEntry(
            iteration=self.iteration,
            added_labels=[r["label"] for r in new_reactions],
            reason=reason,
            score_before=score_before,
        ))

    def record_validation(self, cantera_result: dict, failure_mode: str = ""):
        """Store validation result and update best score."""
        self.last_cantera_result = cantera_result
        score = cantera_result.get("validation_score", 0.0)

        record = ValidationRecord(
            iteration=self.iteration,
            score=score,
            n_successful=cantera_result.get("n_successful", 0),
            n_conditions=cantera_result.get("n_conditions", 4),
            idt_results=cantera_result.get("idt_results", []),
            failure_mode=failure_mode,
        )
        self.validation_history.append(record)

        # Update addition_log with score_after
        if self.addition_log:
            self.addition_log[-1].score_after = score

        if score > self.best_score:
            self.best_score = score
            self.best_mechanism_yaml = self.current_mechanism_yaml
            self.best_reactions_snapshot = list(self.current_reactions)

    def is_done(self) -> bool:
        return (self.best_score >= self.acceptance_threshold
                or self.iteration >= self.max_iters)

    def summary(self) -> str:
        lines = [
            f"Run {self.run_id} | Iteration {self.iteration}/{self.max_iters}",
            f"Best score: {self.best_score:.1%}",
            f"Reactions: {len(self.current_reactions)} | Species: {len(self.current_species)}",
            f"Additions: {len(self.addition_log)}",
        ]
        for entry in self.addition_log:
            lines.append(
                f"  iter {entry.iteration}: +{entry.added_labels} "
                f"(score {entry.score_before:.0%}→{entry.score_after:.0%}) — {entry.reason[:80]}"
            )
        return "\n".join(lines)