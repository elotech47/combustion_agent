"""
CombustionAgent — main agentic loop.

Loop:
  1. PLAN     : Agent reasons about fuel, target conditions, strategy
  2. RETRIEVE : Agent queries DB for relevant reactions/species
  3. BUILD    : Agent assembles Cantera YAML mechanism
  4. VALIDATE : Cantera runs IDT/LFS validation
  5. ANALYZE  : Flux/sensitivity analysis identifies weak points
  6. DECIDE   : Agent decides to ACCEPT, REFINE, or ESCALATE
  7. → back to 2 if REFINE (up to MAX_ITERATIONS)

All tool calls are routed through the LLMOrchestrator.
The agent sees tool results and reasons about them naturally.
"""

import json
import logging
import os
import time
from pathlib import Path
from datetime import datetime

import config
from llm import LLMOrchestrator, LLMResponse
from tools import db_retrieval, cantera_tool, flux_analyzer, literature
from mechanism_builder import dispatch as builder_dispatch

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Tool registry — all tools the agent can call
# ─────────────────────────────────────────────────────────────────────────────

ALL_TOOL_SCHEMAS = (
    db_retrieval.TOOL_SCHEMAS
    + cantera_tool.TOOL_SCHEMAS
    + flux_analyzer.TOOL_SCHEMAS
    + literature.TOOL_SCHEMAS
    + [  # mechanism builder
        {
            "name": "build_mechanism",
            "description": (
                "Assemble a Cantera YAML mechanism from a list of reaction dicts. "
                "Returns the YAML string and build report. "
                "IMPORTANT: Pass the full `reactions` list from get_reactions_for_fuel, "
                "not individual reactions. The tool handles kinetics parsing internally."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "reactions":      {"type": "array", "items": {"type": "object"}},
                    "mechanism_name": {"type": "string"},
                    "description":    {"type": "string"},
                    "extra_species":  {"type": "array", "items": {"type": "string"}},
                },
                "required": ["reactions"],
            },
        }
    ]
)


def dispatch_tool(tool_name: str, args: dict) -> str:
    """Route tool call to the correct module. Logs full input/output at DEBUG level."""
    db_tools      = {s["name"] for s in db_retrieval.TOOL_SCHEMAS}
    cantera_tools = {s["name"] for s in cantera_tool.TOOL_SCHEMAS}
    flux_tools    = {s["name"] for s in flux_analyzer.TOOL_SCHEMAS}
    lit_tools     = {s["name"] for s in literature.TOOL_SCHEMAS}

    # Trace: log args (truncate large fields)
    def _trace_args(a: dict) -> str:
        parts = []
        for k, v in a.items():
            if isinstance(v, str) and len(v) > 200:
                parts.append(f"{k}=<str len={len(v)}>")
            elif isinstance(v, list) and len(v) > 10:
                parts.append(f"{k}=<list len={len(v)}>")
            else:
                parts.append(f"{k}={v!r}")
        return ", ".join(parts)

    logger.debug(f"TOOL CALL ► {tool_name}({_trace_args(args)})")

    if tool_name in db_tools:
        result_str = db_retrieval.dispatch(tool_name, args)
    elif tool_name in cantera_tools:
        result_str = cantera_tool.dispatch(tool_name, args)
    elif tool_name in flux_tools:
        result_str = flux_analyzer.dispatch(tool_name, args)
    elif tool_name in lit_tools:
        result_str = literature.dispatch(tool_name, args)
    elif tool_name == "build_mechanism":
        result_str = builder_dispatch(tool_name, args)
    else:
        result_str = json.dumps({"error": f"Unknown tool: {tool_name}"})

    # Trace: log result (truncate yaml/large fields)
    try:
        result_preview = json.loads(result_str)
        # Redact large fields for log readability
        for field in ("mechanism_yaml", "adjacency_list", "reactions"):
            if field in result_preview:
                v = result_preview[field]
                size = len(v) if isinstance(v, (str, list)) else "?"
                result_preview[field] = f"<{field} len={size}>"
        logger.debug(f"TOOL RESULT ◄ {tool_name}: {json.dumps(result_preview)[:500]}")
    except Exception:
        logger.debug(f"TOOL RESULT ◄ {tool_name}: {result_str[:300]}")

    return result_str


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert combustion chemist AI agent. Your task is to autonomously generate, validate, and refine chemical kinetic mechanisms for combustion applications.

## CRITICAL RULES — READ CAREFULLY

**NEVER fabricate validation results.** You do NOT know the ignition delay or flame speed of any mechanism you generate. These MUST be computed by calling the `validate_mechanism` tool. Any numbers you state without tool calls are hallucinations and will be wrong.

**ALWAYS follow this exact sequence of tool calls:**
1. `get_reactions_for_fuel` → retrieve reactions from database
2. `build_mechanism` → assemble Cantera YAML (pass the reactions list from step 1)
3. `validate_mechanism` → run actual Cantera simulations (pass the mechanism_yaml from step 2)
4. `sensitivity_analysis` → diagnose if score < 0.75 (pass the mechanism_yaml from step 3)
5. Back to step 1 with targeted additions, then rebuild → revalidate → repeat

**You MUST call `build_mechanism` and `validate_mechanism` in every iteration.**
**You MUST NOT write a final summary until `validate_mechanism` has been called at least once.**
**The validation score in your summary MUST come from a tool result, not your knowledge.**

## Available tools
- `get_reactions_for_fuel` — query 35,000+ RMG reactions by species list
- `search_reactions_by_label` — find specific reactions by label pattern
- `build_mechanism` — assemble Cantera YAML from a reactions list
- `validate_mechanism` — run real Cantera IDT + LFS simulations
- `sensitivity_analysis` — identify which reactions control ignition
- `identify_missing_reactions` — find structurally disconnected species
- `rate_of_production` — dominant production/consumption pathways
- `get_reference_mechanism_info` — published reference mechanism info
- `search_literature` — web search for rate constants

## Workflow detail

**RETRIEVE:** Call `get_reactions_for_fuel` with:
  fuel_species=["H2","O2","H","O","OH","H2O","HO2","H2O2"]
  library_preference=["GRI-Mech3.0","H2-O2"]

**BUILD:** Call `build_mechanism` with:
  - The `reactions` list from the retrieval result (reactions field)
  - extra_species=["N2","AR"]
  This returns `mechanism_yaml` — you MUST pass this exact string to validate.

**VALIDATE:** Call `validate_mechanism` with:
  - `mechanism_yaml`: the yaml string from build_mechanism result
  - `compute_lfs`: false (faster for first pass)
  This returns a real `validation_score` from Cantera. Use THIS score.

**DIAGNOSE (if score < 0.75):** Call `sensitivity_analysis` with the same mechanism_yaml.
  Look at top_sensitive_reactions. Then search for those specific reactions with
  `search_reactions_by_label` and add them to the next build call.

**REFINE:** Call `get_reactions_for_fuel` again with a broader or targeted species list,
  merge with previous reactions (avoid duplicates), rebuild, revalidate.

## Rules
- N2 and AR are bath gases — always include in extra_species
- 10-25 reactions is appropriate scope for H2/O2 POC
- Pass the FULL reactions list (not just new ones) to build_mechanism each iteration
- Stop when validation_score >= 0.75 OR after max iterations
- Your final summary validation numbers MUST match what validate_mechanism returned
"""


# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────

class CombustionAgent:
    def __init__(self, backend: str | None = None):
        self.llm        = LLMOrchestrator(backend=backend)
        self.messages:  list[dict] = []
        self.iteration  = 0
        self.history:   list[dict] = []   # per-iteration records
        self.best_mechanism_yaml: str | None = None
        self.best_score: float = 0.0

        # Output dir
        self.output_dir = Path(config.OUTPUT_DIR)
        self.output_dir.mkdir(exist_ok=True)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def run(self, fuel: str = "H2", task: str | None = None) -> dict:
        """
        Main entry point. Runs the full generate→validate→refine loop.

        Args:
            fuel: Target fuel (default H2)
            task: Optional task override (defaults to standard H2/O2 generation)

        Returns:
            Final report dict
        """
        if task is None:
            task = (
                f"Generate a validated H2/O2 combustion kinetic mechanism for fuel={fuel}. "
                f"Target: ignition delay predictions within 30% of reference values across "
                f"T=800-1200K at P=1atm, phi=1.0. "
                f"Use the available tools in sequence: retrieve reactions → build mechanism → "
                f"validate → analyze → refine. "
                f"Maximum {config.MAX_REFINEMENT_ITERATIONS} refinement iterations."
            )

        logger.info(f"Starting CombustionAgent run: fuel={fuel}, run_id={self.run_id}")
        print(f"\n{'='*60}")
        print(f"  CombustionAgent — {fuel} Mechanism Generation")
        print(f"  Run ID: {self.run_id}")
        print(f"  Backend: {self.llm.backend} / {self.llm.model}")
        print(f"{'='*60}\n")

        self.messages = [{"role": "user", "content": task}]

        # State machine to track what the agent must do next
        required_next = "retrieve"   # retrieve → build → validate → (diagnose →) refine/accept

        for i in range(config.MAX_REFINEMENT_ITERATIONS + 2):
            self.iteration = i
            print(f"\n--- Agent Turn {i+1} [{required_next.upper()}] ---")

            response = self._agent_turn()

            if response.content:
                print(f"\n[Agent]: {response.content[:500]}{'...' if len(response.content) > 500 else ''}")

            # No tool calls — agent tried to finish early
            if not response.tool_calls:
                if required_next != "done":
                    # Force continuation — agent short-circuited
                    nudge = self._build_nudge(required_next)
                    logger.warning(f"Agent stopped early at stage '{required_next}'. Nudging: {nudge}")
                    print(f"\n  [Orchestrator nudge → {required_next}]: {nudge}")
                    self.messages.append({"role": "user", "content": nudge})
                    continue
                else:
                    break

            # Execute all tool calls
            tools_called = set()
            for tc in response.tool_calls:
                print(f"\n  → Tool: {tc.name}({self._summarize_args(tc.arguments)})")
                result_str = dispatch_tool(tc.name, tc.arguments)
                result     = json.loads(result_str)

                self._maybe_update_best(tc.name, result)
                self._log_tool_call(tc.name, tc.arguments, result)
                print(f"    ← {self._summarize_result(tc.name, result)}")

                self.messages.append(self.llm.tool_result_message(tc, result_str))
                tools_called.add(tc.name)

            # Advance state machine based on what was actually called
            required_next = self._advance_state(required_next, tools_called)

            if required_next == "done":
                break

        return self._finalize()

    # ──────────────────────────────────────────────────────────────────────────
    # State machine
    # ──────────────────────────────────────────────────────────────────────────

    def _advance_state(self, current: str, tools_called: set) -> str:
        """Advance required_next state based on what tools were actually called."""
        if "get_reactions_for_fuel" in tools_called or "search_reactions_by_label" in tools_called:
            if current == "retrieve":
                return "build"
        if "build_mechanism" in tools_called:
            if current in ("retrieve", "build"):
                return "validate"
        if "validate_mechanism" in tools_called:
            if self.best_score >= 0.75:
                return "done"
            else:
                return "diagnose"
        if "sensitivity_analysis" in tools_called or "identify_missing_reactions" in tools_called:
            return "retrieve"
        return current

    def _build_nudge(self, required_next: str) -> str:
        """Return a specific, actionable nudge to unblock the agent."""
        nudges = {
            "retrieve": (
                "You have not retrieved reactions yet. "
                "Call `get_reactions_for_fuel` with "
                "fuel_species=[\"H2\",\"O2\",\"H\",\"O\",\"OH\",\"H2O\",\"HO2\",\"H2O2\"] "
                "and library_preference=[\"GRI-Mech3.0\"]."
            ),
            "build": (
                "You retrieved reactions but have NOT built the mechanism yet. "
                "Call `build_mechanism` now with the `reactions` list from the previous result "
                "and extra_species=[\"N2\",\"AR\"]. "
                "IMPORTANT: Do not write any IDT or validation numbers yet — "
                "those MUST come from the validate_mechanism tool."
            ),
            "validate": (
                "You built a mechanism but have NOT validated it yet. "
                "Call `validate_mechanism` now with the `mechanism_yaml` string "
                "from the build_mechanism result. Set compute_lfs=false for speed. "
                "IMPORTANT: All validation numbers in your summary MUST come from "
                "this tool's actual output — do not fabricate them."
            ),
            "diagnose": (
                "Validation score is below 0.75. "
                "Call `sensitivity_analysis` with the mechanism_yaml to identify "
                "which reactions to add next. Do not write a final summary yet."
            ),
        }
        return nudges.get(required_next,
            f"Continue to the next required step: {required_next}. "
            "Call the appropriate tool before writing any results."
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Core turn
    # ──────────────────────────────────────────────────────────────────────────

    def _agent_turn(self) -> LLMResponse:
        response = self.llm.chat_with_tools(
            messages=self.messages,
            tools=ALL_TOOL_SCHEMAS,
            system_prompt=SYSTEM_PROMPT,
        )
        # Append assistant message (with tool use blocks if any)
        self.messages.append(self.llm.assistant_tool_use_message(response))
        return response

    # ──────────────────────────────────────────────────────────────────────────
    # Best mechanism tracking
    # ──────────────────────────────────────────────────────────────────────────

    def _maybe_update_best(self, tool_name: str, result: dict):
        if tool_name == "build_mechanism":
            yaml = result.get("mechanism_yaml")
            if yaml:
                self.best_mechanism_yaml = yaml
                logger.info(f"Stored mechanism: {result.get('report',{}).get('n_reactions','?')} reactions")

        if tool_name == "validate_mechanism":
            score = result.get("validation_score", 0.0)
            if score > self.best_score:
                self.best_score = score
                logger.info(f"New best validation score: {score:.3f}")

        if tool_name == "build_mechanism":
            yaml = result.get("mechanism_yaml")
            if yaml:
                self.best_mechanism_yaml = yaml

    # ──────────────────────────────────────────────────────────────────────────
    # Finalize
    # ──────────────────────────────────────────────────────────────────────────

    def _finalize(self) -> dict:
        print(f"\n{'='*60}")
        print(f"  Run Complete — Best Score: {self.best_score:.1%}")
        print(f"{'='*60}")

        # Ask agent for final summary — inject actual scores to prevent hallucination
        actual_results = (
            f"ACTUAL RESULTS FROM TOOL CALLS:\n"
            f"- Best validation score: {self.best_score:.1%}\n"
            f"- Total iterations: {self.iteration + 1}\n"
            f"- Mechanism stored: {'YES' if self.best_mechanism_yaml else 'NO'}\n\n"
            f"Using ONLY the actual tool call results above and in the conversation history, "
            f"provide a final summary covering: "
            f"(1) mechanism species and reaction count, "
            f"(2) actual validation scores and IDT results from the tool outputs, "
            f"(3) what worked and what to improve, "
            f"(4) comparison to Li et al. 2004 H2/O2 reference. "
            f"Do NOT invent any numbers not present in the tool results."
        )
        self.messages.append({"role": "user", "content": actual_results})
        final_response = self.llm.chat(
            messages=self.messages,
            system_prompt=SYSTEM_PROMPT,
        )
        print(f"\n[Final Summary]:\n{final_response}")

        # Save outputs
        report = {
            "run_id":        self.run_id,
            "backend":       self.llm.backend,
            "model":         self.llm.model,
            "best_score":    self.best_score,
            "iterations":    self.iteration + 1,
            "tool_call_log": self.history,
            "final_summary": final_response,
        }

        self._save_outputs(report)
        return report

    def _save_outputs(self, report: dict):
        """Save mechanism YAML and run report."""
        run_dir = self.output_dir / self.run_id
        run_dir.mkdir(exist_ok=True)

        # Save mechanism
        if self.best_mechanism_yaml:
            mech_path = run_dir / "mechanism.yaml"
            mech_path.write_text(self.best_mechanism_yaml)
            print(f"\nMechanism saved: {mech_path}")

        # Save report
        report_path = run_dir / "report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved:    {report_path}")

    # ──────────────────────────────────────────────────────────────────────────
    # Logging helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _log_tool_call(self, name: str, args: dict, result: dict):
        self.history.append({
            "iteration":  self.iteration,
            "tool":       name,
            "args_keys":  list(args.keys()),
            "result_summary": self._summarize_result(name, result),
            "timestamp":  time.time(),
        })

    def _summarize_args(self, args: dict) -> str:
        """One-line arg summary for printing."""
        parts = []
        for k, v in args.items():
            if isinstance(v, str) and len(v) > 60:
                parts.append(f"{k}=<{len(v)} chars>")
            elif isinstance(v, list) and len(v) > 5:
                parts.append(f"{k}=[{len(v)} items]")
            else:
                parts.append(f"{k}={v!r}")
        return ", ".join(parts[:4])

    def _summarize_result(self, tool_name: str, result: dict) -> str:
        if "error" in result:
            return f"ERROR: {result['error']}"
        if tool_name == "validate_mechanism":
            score = result.get("validation_score", "?")
            n_ok  = result.get("n_successful", "?")
            n_tot = result.get("n_conditions", "?")
            passed = "✓ PASSED" if result.get("passed") else "✗ FAILED"
            return f"Score={score:.2%} ({n_ok}/{n_tot} conditions) {passed}"
        if tool_name == "build_mechanism":
            r = result.get("report", {})
            skipped = r.get("skipped_reactions", [])
            skip_summary = ""
            if skipped:
                # Group by reason
                from collections import Counter
                reasons = Counter(s.get("reason", "unknown") for s in skipped)
                skip_summary = " | skipped: " + ", ".join(f"{v}×'{k}'" for k, v in reasons.most_common(3))
                # Print first few skipped for diagnosis
                for s in skipped[:3]:
                    logger.debug(f"  Skipped: {s.get('label','?')!r} — {s.get('reason','?')}")
            return (f"{r.get('n_species','?')} species, {r.get('n_reactions','?')} reactions, "
                    f"valid={r.get('valid')}{skip_summary}")
        if tool_name == "get_reactions_for_fuel":
            return f"{result.get('count', '?')} reactions returned"
        if tool_name == "sensitivity_analysis":
            top = result.get("top_sensitive_reactions", [])
            if top:
                return f"Top reaction: {top[0]['reaction']} (sens={top[0]['sensitivity']:.3f})"
        if tool_name == "get_db_stats":
            return f"DB: {result.get('total_reactions','?')} reactions, {result.get('libraries','?')} libraries"
        return str(result)[:120]