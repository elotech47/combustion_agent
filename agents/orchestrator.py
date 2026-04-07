"""
MultiAgentOrchestrator.

Coordinates the full generate → validate → diagnose → select → refine loop.

Each subagent has its own LLMOrchestrator instance.
The orchestrator owns OrchestratorState and is the only entity that mutates it.
Subagents receive context slices, return structured outputs, never touch state directly.

Loop per iteration:
  1. DB Search     → refresh candidate pool (deterministic, no LLM)
  2. Selector      → pick 1-3 reactions from pool (LLM with full chemistry context)
  3. Builder       → assemble YAML (deterministic)
  4. Validator     → run Cantera (deterministic)
  5. Diagnostician → interpret results (LLM with cantera context)
  6. Orchestrator  → decide: accept / refine / escalate
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import config
from llm import LLMOrchestrator
from agents.state import OrchestratorState, DiagnosticianReport
from agents import db_search, selector as selector_agent, diagnostician as diag_agent
from mechanism_builder import build_mechanism_yaml, dispatch as builder_dispatch
from tools import cantera_tool, flux_analyzer
from tools import family_tool

logger = logging.getLogger(__name__)


class MultiAgentOrchestrator:
    """
    Top-level coordinator. Owns state, creates subagent LLMs, runs the loop.
    """

    def __init__(
        self,
        backend:     str | None = None,
        diag_model:  str | None = None,
        sel_model:   str | None = None,
    ):
        backend = backend or config.LLM_BACKEND

        # Each subagent gets its own LLM instance
        # (can point to different models if desired)
        self.llm_selector     = LLMOrchestrator(backend=backend)
        self.llm_diagnostician = LLMOrchestrator(backend=backend)

        # Override models if specified
        if diag_model:
            self.llm_diagnostician.model = diag_model
        if sel_model:
            self.llm_selector.model = sel_model

        self.output_dir = Path(config.OUTPUT_DIR)
        self.output_dir.mkdir(exist_ok=True)

        logger.info(
            f"MultiAgentOrchestrator initialized | "
            f"selector={self.llm_selector.model} | "
            f"diagnostician={self.llm_diagnostician.model}"
        )

    def run(self, fuel: str = "H2", max_iters: int | None = None) -> dict:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        state  = OrchestratorState(
            fuel=fuel,
            run_id=run_id,
            max_iters=max_iters or config.MAX_REFINEMENT_ITERATIONS,
        )

        self._print_header(state)

        while not state.is_done():
            self._run_iteration(state)
            state.iteration += 1

        return self._finalize(state)

    # ─────────────────────────────────────────────────────────────────────────
    # Single iteration
    # ─────────────────────────────────────────────────────────────────────────

    def _run_iteration(self, state: OrchestratorState):
        it = state.iteration
        print(f"\n{'─'*60}")
        print(f"  ITERATION {it} | current={len(state.current_reactions)} rxns | best={state.best_score:.1%}")
        print(f"{'─'*60}")

        # ── Step 1: DB Search ────────────────────────────────────────────────
        print(f"\n[1/5] DB Search → refreshing candidate pool")
        # On refinement iterations, add diagnostician targets to search.
        # Guard: only accept species that are already known to the fuel system.
        # This prevents the diagnostician from pulling in out-of-scope species
        # (e.g. carbon species appearing in a Cantera error for an H2/O2 run).
        if state.diagnostician_report:
            _allowed = (
                {s.upper() for s in state.target_species}
                | {s.upper() for s in state.current_species}
                | {s.upper() for s in state.bath_gases}
                | {"N2", "AR", "HE", "M"}
            )
            extra = state.diagnostician_report.target_species
            for sp in extra:
                if sp.upper() not in _allowed:
                    logger.warning(
                        f"[Orchestrator] Ignoring out-of-scope target species from diagnostician: {sp!r}"
                    )
                    continue
                if sp not in state.target_species:
                    state.target_species.append(sp)

        candidates = db_search.run(state)
        state.candidate_pool = candidates
        print(f"      Pool: {len(candidates)} candidates "
              f"({sum(1 for r in candidates if _is_third_body(r))} third-body)")

        if not candidates:
            print("      WARNING: Empty candidate pool — skipping iteration")
            return

        # ── Step 2: Chemistry Selector ───────────────────────────────────────
        print(f"\n[2/5] Chemistry Selector → choosing reactions to add")
        sel_output = selector_agent.run(state, self.llm_selector)

        if not sel_output.selected:
            print("      Selector returned nothing valid — skipping iteration")
            return

        selected_labels = [r["label"] for r in sel_output.selected]
        print(f"      Selected: {selected_labels}")
        print(f"      Reason:   {sel_output.reasoning[:200]}")

        # Orchestrator adds to state (only mutator)
        state.add_reactions(sel_output.selected, reason=sel_output.reasoning[:300])

        # Update target species for next search
        if sel_output.next_target_species:
            for sp in sel_output.next_target_species:
                if sp not in state.target_species:
                    state.target_species.append(sp)

        # ── Step 3: Build Mechanism ──────────────────────────────────────────
        print(f"\n[3/5] Mechanism Builder → assembling YAML")
        result_str = builder_dispatch("build_mechanism", {
            "reactions":      state.current_reactions,
            "mechanism_name": f"CombustionAgent {state.fuel} iter={it}",
            "description":    f"Multi-agent iteration {it}",
            "extra_species":  state.bath_gases,
        })
        build_result = json.loads(result_str)

        if "error" in build_result:
            print(f"      BUILD ERROR: {build_result['error']}")
            return

        report = build_result.get("report", {})
        yaml   = build_result.get("mechanism_yaml", "")
        print(f"      Built: {report.get('n_species')} species, "
              f"{report.get('n_reactions')} reactions | valid={report.get('valid')}")

        skipped = report.get("skipped_reactions", [])
        if skipped:
            from collections import Counter
            reasons = Counter(s.get("reason", "?") for s in skipped)
            print(f"      Skipped: {dict(reasons)}")

        if not report.get("valid") or not yaml:
            print("      Mechanism invalid — skipping validation")
            return

        state.current_mechanism_yaml = yaml
        self._save_intermediate(state, yaml, it)

        # ── Step 4: Cantera Validation ───────────────────────────────────────
        print(f"\n[4/5] Cantera Validator → running IDT simulations")
        val_result = cantera_tool.validate_mechanism(
            yaml,
            conditions=state.target_conditions,
            compute_lfs=False,
        )
        score = val_result.get("validation_score", 0.0)
        print(f"      Score: {score:.1%} "
              f"({val_result.get('n_successful')}/{val_result.get('n_conditions')} conditions)")
        print(f"      {val_result.get('summary', '').strip()}")

        # ── Step 4b: Flux Analysis (when mechanism partially works) ─────────
        if score > 0 and yaml:
            print(f"\n[4b] Flux Analysis → sensitivity + structural check")
            try:
                sens = flux_analyzer.sensitivity_analysis(yaml, T=1000, top_n=10)
                isolated = flux_analyzer.identify_missing_reactions(
                    state.current_species, state.current_reactions
                )
                state.flux_analysis = {
                    "top_sensitive_reactions": sens.get("top_sensitive_reactions", []),
                    "isolated_species":        isolated.get("isolated_species", []),
                    "most_promoting":          sens.get("most_promoting"),
                    "most_inhibiting":         sens.get("most_inhibiting"),
                }
                if state.flux_analysis["top_sensitive_reactions"]:
                    top3 = state.flux_analysis["top_sensitive_reactions"][:3]
                    print(f"      Top sensitive: {[r['reaction'] for r in top3]}")
                if state.flux_analysis["isolated_species"]:
                    print(f"      Isolated species: {state.flux_analysis['isolated_species']}")

                # Family coverage check — tells diagnostician which reaction families are missing
                try:
                    labels   = [r["label"] for r in state.current_reactions]
                    ktypes   = [r.get("kinetics_type", "Arrhenius")
                                for r in state.current_reactions]
                    coverage = family_tool.check_mechanism_coverage(
                        state.current_species, labels, ktypes
                    )
                    state.flux_analysis["family_coverage"] = coverage
                    missing_fam = coverage.get("missing_families", [])
                    if missing_fam:
                        print(f"      Missing families: {missing_fam[:5]}")
                except Exception as e:
                    logger.warning(f"Family coverage check failed (non-fatal): {e}")

            except Exception as e:
                logger.warning(f"Flux analysis failed (non-fatal): {e}")

        # ── Step 5: Diagnostician ────────────────────────────────────────────
        print(f"\n[5/5] Diagnostician → interpreting results")
        prev_best = state.best_score
        state.record_validation(val_result)

        # ── Rollback if score regressed significantly ───────────────────────────
        # Require TWO consecutive drops below best before reverting.
        # A single bad iteration may be a necessary stepping stone (e.g., adding
        # H2O2 chemistry temporarily destabilises until decomposition is also added).
        recent_scores = [v.score for v in state.validation_history[-2:]]
        consecutive_drop = (
            len(recent_scores) >= 2
            and all(s < prev_best - 0.05 for s in recent_scores)
        )
        if consecutive_drop and state.best_reactions_snapshot:
            harmful = [
                r["label"] for r in state.current_reactions
                if r["label"] not in {x["label"] for x in state.best_reactions_snapshot}
            ]
            print(f"\n  ⚠ ROLLBACK: 2 consecutive drops below best {prev_best:.1%}")
            print(f"    Reverting to best state and blacklisting: {harmful}")
            state.current_reactions = list(state.best_reactions_snapshot)
            state.already_tried_labels.update(harmful)
            state.current_mechanism_yaml = state.best_mechanism_yaml

        if state.is_done():
            print(f"      Score {score:.1%} ≥ threshold — ACCEPTED")
            return

        diag_report = diag_agent.run(state, self.llm_diagnostician)
        state.diagnostician_report = diag_report
        state.record_validation(val_result, failure_mode=diag_report.failure_mode)

        print(f"      Failure mode: {diag_report.failure_mode}")
        print(f"      Missing:      {diag_report.missing_pathways}")
        print(f"      Targets next: {diag_report.target_species}")
        print(f"      Severity:     {diag_report.severity}")

        # Print full state summary at end of iteration
        print(f"\n{state.summary()}")

    # ─────────────────────────────────────────────────────────────────────────
    # Finalize
    # ─────────────────────────────────────────────────────────────────────────

    def _finalize(self, state: OrchestratorState) -> dict:
        run_dir = self.output_dir / state.run_id
        run_dir.mkdir(exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  COMPLETE | Best score: {state.best_score:.1%}")
        print(f"  Mechanism: {len(state.current_reactions)} reactions | "
              f"{len(state.current_species)} species")
        print(f"{'='*60}")
        print(f"\n{state.summary()}")

        # Save best mechanism
        if state.best_mechanism_yaml:
            mech_path = run_dir / "mechanism_best.yaml"
            mech_path.write_text(state.best_mechanism_yaml)
            print(f"\nMechanism saved: {mech_path}")

        # Save full report
        report = {
            "run_id":       state.run_id,
            "fuel":         state.fuel,
            "best_score":   state.best_score,
            "iterations":   state.iteration,
            "n_reactions":  len(state.current_reactions),
            "n_species":    len(state.current_species),
            "species":      state.current_species,
            "reactions":    [r["label"] for r in state.current_reactions],
            "addition_log": [
                {
                    "iteration":    e.iteration,
                    "added":        e.added_labels,
                    "reason":       e.reason,
                    "score_before": e.score_before,
                    "score_after":  e.score_after,
                }
                for e in state.addition_log
            ],
            "validation_history": [
                {
                    "iteration":    v.iteration,
                    "score":        v.score,
                    "failure_mode": v.failure_mode,
                }
                for v in state.validation_history
            ],
            "selector_backend":     self.llm_selector.backend,
            "selector_model":       self.llm_selector.model,
            "diagnostician_backend": self.llm_diagnostician.backend,
            "diagnostician_model":  self.llm_diagnostician.model,
        }

        report_path = run_dir / "report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved:    {report_path}")

        return report

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _save_intermediate(self, state: OrchestratorState, yaml: str, iteration: int):
        run_dir = self.output_dir / state.run_id
        run_dir.mkdir(exist_ok=True)
        path = run_dir / f"mechanism_iter{iteration:02d}.yaml"
        path.write_text(yaml)
        logger.debug(f"Intermediate mechanism saved: {path}")

    def _print_header(self, state: OrchestratorState):
        print(f"\n{'='*60}")
        print(f"  CombustionAgent — Multi-Agent Mode")
        print(f"  Fuel: {state.fuel} | Run: {state.run_id}")
        print(f"  Selector:     {self.llm_selector.backend}/{self.llm_selector.model}")
        print(f"  Diagnostician: {self.llm_diagnostician.backend}/{self.llm_diagnostician.model}")
        print(f"  Max iters: {state.max_iters} | Threshold: {state.acceptance_threshold:.0%}")
        print(f"{'='*60}")


def _is_third_body(rxn: dict) -> bool:
    """Use kinetics_type as authoritative signal; fall back to label string."""
    kt = rxn.get("kinetics_type", "")
    if kt in ("ThirdBody", "Troe", "Lindemann", "PDepArrhenius", "MultiPDepArrhenius"):
        return True
    label_up = rxn.get("label", "").upper()
    return "(+M)" in label_up or "+ M " in label_up or label_up.strip().endswith("+M")