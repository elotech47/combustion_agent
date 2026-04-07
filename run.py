"""
CombustionAgent — entry point.

Usage:
    python run.py
    python run.py --fuel H2 --backend claude
    python run.py --fuel H2 --backend openai
    python run.py --fuel H2 --backend local
    python run.py --fuel H2 --backend openrouter --model anthropic/claude-opus-4-5

Environment variables (set before running):
    ANTHROPIC_API_KEY   = sk-ant-...
    OPENAI_API_KEY      = sk-...
    OPENROUTER_API_KEY  = sk-or-...
    LOCAL_BASE_URL      = http://localhost:11434/v1  (for Ollama/vLLM)

    REACTIONS_DB_PATH   = /path/to/reactions_database.json
    SPECIES_DB_PATH     = /path/to/species_database.json
    MAX_ITERATIONS      = 5
    OUTPUT_DIR          = output
"""

import argparse
import logging
import os
import sys

def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("cantera").setLevel(logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser(
        description="CombustionAgent — Agentic Mechanism Generation POC"
    )
    parser.add_argument(
        "--fuel", default="H2",
        help="Target fuel (default: H2)"
    )
    parser.add_argument(
        "--backend", default=None,
        choices=["claude", "openai", "openrouter", "local"],
        help="LLM backend (overrides LLM_BACKEND env var)"
    )
    parser.add_argument(
        "--model", default=None,
        help="Model name override (e.g. gpt-4o, claude-opus-4-5)"
    )
    parser.add_argument(
        "--reactions-db", default=None,
        help="Path to reactions_database.json"
    )
    parser.add_argument(
        "--species-db", default=None,
        help="Path to species_database.json"
    )
    parser.add_argument(
        "--max-iter", type=int, default=None,
        help="Max refinement iterations"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory for mechanism + reports"
    )
    parser.add_argument(
        "--multi-agent", action="store_true", default=True,
        help="Use multi-agent loop (default: True)"
    )
    parser.add_argument(
        "--single-agent", action="store_true", default=False,
        help="Use original single-agent loop (legacy)"
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log_level)

    # Apply CLI overrides to config
    import config
    if args.backend:
        config.LLM_BACKEND = args.backend
    if args.model:
        config.MODELS[config.LLM_BACKEND] = args.model
    if args.reactions_db:
        config.REACTIONS_DB_PATH = args.reactions_db
    if args.species_db:
        config.SPECIES_DB_PATH = args.species_db
    if args.max_iter:
        config.MAX_REFINEMENT_ITERATIONS = args.max_iter
    if args.output_dir:
        config.OUTPUT_DIR = args.output_dir

    # Validate API key is set
    backend = config.LLM_BACKEND
    key_map = {
        "claude":     ("ANTHROPIC_API_KEY",  config.ANTHROPIC_API_KEY),
        "openai":     ("OPENAI_API_KEY",     config.OPENAI_API_KEY),
        "openrouter": ("OPENROUTER_API_KEY", config.OPENROUTER_API_KEY),
        "local":      (None, "local"),  # no key needed
    }
    env_name, key_val = key_map.get(backend, (None, None))
    if env_name and not key_val:
        print(f"\n[ERROR] {env_name} is not set.")
        print(f"  Set it with: export {env_name}=your_key_here")
        sys.exit(1)

    # Validate DB paths
    from pathlib import Path
    for label, path in [("Reactions DB", config.REACTIONS_DB_PATH),
                         ("Species DB",   config.SPECIES_DB_PATH)]:
        if not Path(path).exists():
            print(f"\n[ERROR] {label} not found: {path}")
            print(f"  Set with --reactions-db / --species-db flags or env vars.")
            sys.exit(1)

    print(f"\nConfiguration:")
    print(f"  Fuel:         {args.fuel}")
    print(f"  Backend:      {backend} / {config.MODELS[backend]}")
    print(f"  Reactions DB: {config.REACTIONS_DB_PATH}")
    print(f"  Species DB:   {config.SPECIES_DB_PATH}")
    print(f"  Max iter:     {config.MAX_REFINEMENT_ITERATIONS}")
    print(f"  Output dir:   {config.OUTPUT_DIR}")

    use_multi = not args.single_agent
    print(f"  Mode: {'multi-agent' if use_multi else 'single-agent (legacy)'}")

    if use_multi:
        from agents.orchestrator import MultiAgentOrchestrator
        agent  = MultiAgentOrchestrator(backend=backend)
        report = agent.run(fuel=args.fuel, max_iters=config.MAX_REFINEMENT_ITERATIONS)
    else:
        from agent import CombustionAgent
        agent  = CombustionAgent(backend=backend)
        report = agent.run(fuel=args.fuel)

    print(f"✓ Done. Best validation score: {report['best_score']:.1%}")


if __name__ == "__main__":
    main()