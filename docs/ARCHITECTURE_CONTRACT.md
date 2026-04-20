# Architecture Contract

This repo is a pedagogical Layer 1 repo for agentic finance research intelligence.

## Repo Identity

- Layer 1 only: retrieval, bounded synthesis, and structured handoff.
- Public or demo data only.
- Human review required for any downstream decision.
- Research outputs are handoffs for human review, not PM action notes.

## Canonical Pattern

- `examples/example8.py` is canonical because it encodes the repo's intended end-state contract in code.
- The flow is explicit and bounded:
  - factual evidence packet
  - bounded research read
  - open questions / diligence gaps
- New work in this repo should align to that pattern rather than expanding scope.

## Example Roles

- `examples/example0_setup_check.py` through `examples/example6.py` are learning primitives and scaffolding.
- `examples/example7.py` is the bridge example for staged multi-agent research and coordination.
- `examples/example9.py` is an adjacent MCP issuer-intelligence demo, useful for public-source retrieval but not the central contract.

## Shared Helpers

- `examples/finance_tools.py` is the shared read-only market, fundamentals, analyst, and news retrieval helper module.
- `examples/news_filter.py` provides news scoring, filtering, normalization, and selection helpers for public-source quality control.

## Intentionally Not Done Here

- No backtesting, portfolio construction, optimization, execution, monitoring, or risk automation.
- No data-loading pipeline or feature-engineering layer.
- No autonomous portfolio action or private-data workflow.
- No framework abstraction beyond the example-level code already present.
- No PM action notes or investment instructions.
