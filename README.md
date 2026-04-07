# CombustionAgent POC

> Agentic AI for automated combustion kinetic mechanism generation, validation, and refinement.  
> Proof-of-concept

## What This Does

An LLM-based agent that autonomously:
1. **Retrieves** reactions from a 35,000-reaction RMG database
2. **Builds** a Cantera YAML mechanism
3. **Validates** it via 0D ignition delay + 1D laminar flame speed simulations
4. **Analyzes** failure modes via sensitivity/flux analysis
5. **Refines** the mechanism by adding targeted reactions
6. **Reports** final mechanism quality vs. reference benchmarks

Target for POC: H2/O2 mechanism generation with ignition delay within 30% of reference.

## Project Structure

```
combustion_agent/
├── config.py              # Backend selection, API keys, DB paths
├── llm.py                 # LLMOrchestrator (Claude / OpenAI / OpenRouter / Local)
├── agent.py               # Main agentic loop
├── mechanism_builder.py   # Assembles Cantera YAML from DB reactions
├── tools/
│   ├── db_retrieval.py    # Queries reactions_database.json + species_database.json
│   ├── cantera_tool.py    # IDT and LFS via Cantera
│   ├── flux_analyzer.py   # Sensitivity + ROP analysis
│   └── literature.py      # Web search (NIST, DuckDuckGo)
└── run.py                 # CLI entry point
```

## Setup

### 1. Install dependencies

```bash
pip install anthropic openai cantera numpy requests
```

### 2. Place your database files

```bash
# Either in the combustion_agent/ directory (default):
cp /path/to/reactions_database.json ./
cp /path/to/species_database.json ./

# Or specify paths via flags:
python run.py --reactions-db /path/to/reactions_database.json \
              --species-db /path/to/species_database.json
```

### 3. Set API key

```bash
# Claude (recommended)
export ANTHROPIC_API_KEY=sk-ant-...

# OpenAI
export OPENAI_API_KEY=sk-...

# OpenRouter
export OPENROUTER_API_KEY=sk-or-...

# Local (Ollama / vLLM) — no key needed
export LOCAL_BASE_URL=http://localhost:11434/v1
export LOCAL_MODEL=qwen2.5:72b   # or whatever you're serving
```

## Usage

```bash
# Default: H2/O2, Claude backend
python run.py

# Specify backend
python run.py --backend openai
python run.py --backend local
python run.py --backend openrouter --model anthropic/claude-opus-4-5

# Override DB paths
python run.py --reactions-db /data/reactions_database.json \
              --species-db /data/species_database.json

# Control iterations
python run.py --max-iter 3

# Debug mode
python run.py --log-level DEBUG
```

## Outputs

Each run creates `output/<run_id>/`:
- `mechanism.yaml` — best Cantera mechanism generated
- `report.json`    — full run report with tool call log, scores, final summary

## Agent Loop

```
User Task
    │
    ▼
┌─────────────────────────────────────────────┐
│  PLAN: Identify species, conditions          │
│    ↓                                         │
│  RETRIEVE: get_reactions_for_fuel()          │
│    ↓                                         │
│  BUILD: build_mechanism()                    │
│    ↓                                         │
│  VALIDATE: validate_mechanism() via Cantera  │
│    ↓                                         │
│  ANALYZE: sensitivity_analysis()  ←──────┐  │
│    ↓                                      │  │
│  DECIDE: score ≥ 0.75?                   │  │
│    ├─ YES → ACCEPT + report              │  │
│    └─ NO  → add reactions ───────────────┘  │
└─────────────────────────────────────────────┘
```

## Tools Available to the Agent

| Tool | Module | Description |
|------|--------|-------------|
| `get_reactions_for_fuel` | db_retrieval | Query RMG DB for fuel-relevant reactions |
| `get_species_info` | db_retrieval | Look up species adjacency list |
| `search_reactions_by_label` | db_retrieval | Regex search on reaction labels |
| `list_available_libraries` | db_retrieval | List all kinetics libraries |
| `build_mechanism` | mechanism_builder | Assemble Cantera YAML |
| `validate_mechanism` | cantera_tool | Run IDT + LFS validation |
| `compute_ignition_delay` | cantera_tool | Single T/P/phi IDT run |
| `sensitivity_analysis` | flux_analyzer | dIDT/dA_i per reaction |
| `rate_of_production` | flux_analyzer | Dominant pathways per species |
| `identify_missing_reactions` | flux_analyzer | Find isolated species |
| `search_literature` | literature | DuckDuckGo search |
| `search_nist_kinetics` | literature | NIST kinetics DB query |
| `get_reference_mechanism_info` | literature | Published mechanism registry |

## LLM Backends

| Backend | Config | Notes |
|---------|--------|-------|
| `claude` | `ANTHROPIC_API_KEY` | Recommended — best reasoning for chemistry |
| `openai` | `OPENAI_API_KEY` | GPT-4o works well |
| `openrouter` | `OPENROUTER_API_KEY` | Access any model via one API |
| `local` | `LOCAL_BASE_URL` | Qwen2.5-72B, Llama-3, etc. via vLLM/Ollama |

## Extending

### Add a new tool
1. Create tool function in `tools/` with a `TOOL_SCHEMAS` list and `dispatch()` function
2. Import and add schemas to `ALL_TOOL_SCHEMAS` in `agent.py`
3. Add routing in `dispatch_tool()` in `agent.py`

### Add a new fuel
Update `config.py` `VALIDATION_TARGETS` with reference IDT values for the new fuel.  
Add NASA7 thermodynamic data to `mechanism_builder.py` for any new species.

## Research Context

This POC demonstrates the core loop proposed in our AWS Agentic AI grant application:  
*"Agentic AI for Accelerating Scientific Discovery in Combustion Chemistry"*

The full research system extends this with:
- Uncertainty quantification on generated rate constants
- Multi-agent coordination (separate generation, validation, and reduction agents)
- Broader fuel coverage (NH3/H2 blends, sustainable aviation fuels)
- Integration with the adaptive ODE solver selection framework (RL-based CVODE/QSS switching)
