# Technical Debt and Cleanup Backlog

This backlog treats [`examples/example10/`](../examples/example10/) as the main maintainable system and [`examples/example1.py`](../examples/example1.py) through [`examples/example9.py`](../examples/example9.py) as tutorial/demo assets.

It focuses on:

- architecture
- implementation
- docs
- environment
- testing
- output credibility
- module boundaries

## Current Observations

| Observation | Evidence |
| --- | --- |
| The only real multi-module architecture lives in `example10`. | [`examples/example10/`](../examples/example10/) |
| The repo currently has no test files. | No `tests/`, `test_*.py`, or `*_test.py` files were found. |
| The setup check is too shallow to validate the repository. | [`examples/example0_setup_check.py`](../examples/example0_setup_check.py) imports only `agno.agent.Agent`. |
| `example10` already captures provenance internally, but does not fully surface it in the final output. | [`examples/example10/adapters/mcp_web_adapter.py`](../examples/example10/adapters/mcp_web_adapter.py), [`examples/example10/adapters/mcp_macro_adapter.py`](../examples/example10/adapters/mcp_macro_adapter.py), [`examples/example10/services/formatter.py`](../examples/example10/services/formatter.py) |

## Fix Now

| Priority | Area | Item | Why it matters | Blocks freeze |
| --- | --- | --- | --- | --- |
| 1 | testing | Add deterministic automated tests and a CI smoke path for `example10`. Cover brief generation, web fallback/live normalization, macro parsing, market fallback behavior, portfolio synthesis, and report formatting. | The most complex logic is heuristic and multi-step, but the repo currently has no test suite. Regressions in the MCP adapters, signal gating, or formatter will be invisible until a user notices a bad output. | Yes |
| 2 | output credibility | Make degraded market and macro inputs non-actionable, or gate actionability on explicit live completeness thresholds. | [`MarketAdapter`](../examples/example10/adapters/market_adapter.py) and [`MCPMacroAdapter`](../examples/example10/adapters/mcp_macro_adapter.py) can inject placeholders/fallbacks, and [`PortfolioSynthesisAgent`](../examples/example10/agents/portfolio_synthesis_agent.py) still scores those inputs. [`ResearchOrchestrator`](../examples/example10/services/orchestrator.py) labels such runs as `cautious-tactical`, not strictly research-only. That is too permissive for a freeze if output trust matters. | Yes |
| 3 | output credibility / docs | Surface verifiable provenance in the final report: evidence URLs, timestamps, live vs fallback status, per-indicator macro completeness, and placeholder market tickers. | `Evidence` already stores `url` and `timestamp`, and both adapters already keep detailed call reports. The user-facing formatter currently shows counts, source names, and headlines, but not enough information to independently verify the packet. | Yes |
| 4 | environment / docs | Replace `example0_setup_check.py` with a truthful repository smoke check and update the README to match it. | The README tells users to run `example0` to confirm the environment, but the current script only checks whether `agno` imports. It does not validate OpenAI usage, Tavily, yfinance, FRED, MCP configuration, optional `rich`, or degraded vs live mode expectations. | Yes |
| 5 | docs | Add a short “trust contract” section for `example10`: what live mode means, what degraded mode means, when outputs are research-only, and what should never be treated as deployable. | The repo already carries a disclaimer, but the operational trust boundary is more important than the legal disclaimer. A user should not need to infer the credibility model from code paths. | Yes |

## Next Version

| Priority | Area | Item | Why it matters | Blocks freeze |
| --- | --- | --- | --- | --- |
| 1 | module boundaries | Replace intermediate `dict` payloads with typed models for `web_view`, `market_view`, `macro_view`, and `tactical_state`. | [`ResearchPacket`](../examples/example10/schemas.py) uses dataclasses for some structures, but the most important cross-module contracts are still untyped dictionaries. That makes refactors brittle and pushes correctness into convention. | No |
| 2 | architecture / implementation | Break up [`mcp_web_adapter.py`](../examples/example10/adapters/mcp_web_adapter.py) into smaller components: MCP client, response normalizer, scoring/ranking, extract enrichment, and provenance reporting. | The core web credibility path currently lives in a single large adapter. It is hard to reason about, hard to test in isolation, and easy to regress. | No |
| 3 | implementation | Remove duplicated business rules and formatting logic across the CLI renderer, formatter, macro adapter, and macro MCP server. | Status rendering is duplicated between [`example10.py`](../examples/example10/example10.py) and [`formatter.py`](../examples/example10/services/formatter.py). Macro regime classification is duplicated in multiple places. Drift is likely over time. | No |
| 4 | architecture / module boundaries | Introduce a real package boundary and entry points instead of relying on script-relative import fallbacks inside `examples/`. | The repeated `try/except ImportError` pattern across `example10` is a sign that the repo is balancing package use and script execution at the same time. That is manageable today, but it is not a clean long-term boundary. | No |
| 5 | environment | Split dependencies into clearer extras such as `core`, `live`, `dev`, and `ui`, and make optional dependencies explicit. | `rich` is used by the `example10` CLI but is not declared as a dependency. The repository also mixes tutorial dependencies, live retrieval dependencies, and dev tooling in one flat surface. | No |
| 6 | docs | Document the intended maintenance boundary: which examples are stable references, which are demos, and which modules are allowed to evolve aggressively. | Right now the README explains the progression of examples but not the maintenance policy. That leads to ambiguity about what needs production-like hardening. | No |

## Later / Optional

| Priority | Area | Item | Why it matters | Blocks freeze |
| --- | --- | --- | --- | --- |
| 1 | testing / output credibility | Build a fixture corpus for source-quality ranking, freshness scoring, ticker extraction, and macro regime classification. | This would make the heuristic layer easier to evaluate objectively, but it can follow the basic freeze-protection tests. | No |
| 2 | architecture | Promote run history from a report archive into a queryable evaluation layer only if repeated analysis becomes important. | [`SQLiteStorage`](../examples/example10/services/storage.py) is intentionally simple today. More structure would help analytics later, but it is not urgent. | No |
| 3 | implementation / environment | Consolidate repetitive model-id and environment-variable plumbing across `example7` to `example9`. | Those scripts repeat the same pattern. Cleanup would reduce friction, but it should not take priority over trust and testability in `example10`. | No |
| 4 | docs / tooling | Add a lightweight CLI or export path for saving packets as JSON/Markdown artifacts. | Useful for reproducibility and review, but not required to stabilize the current repository. | No |

## Do Not Touch

| Area | Item | Why it should stay as-is for now |
| --- | --- | --- |
| architecture | Do not try to productionize `example1` to `example9` into a single polished framework. | Those files are most useful as explicit, readable demos. Abstracting them too early will make the repo harder to learn from without solving the real freeze risks. |
| output credibility | Do not remove degraded/offline fallbacks entirely. | The fallbacks are valuable for demos and local exploration. The right move is stricter actionability gating and clearer provenance, not removing offline behavior. |
| environment | Do not replace the simple SQLite run history with a heavier database or ORM abstraction yet. | The current storage need is small and local. More infrastructure would add complexity without improving the highest-risk areas. |
| module boundaries | Do not fold `examples/finance_tools.py` into `example10` until the repository’s product boundary is clearer. | The README already positions it as a compact demo utility layer, not a production SDK. Merging it into the capstone would blur boundaries further. |

## Recommended Freeze Gate

Before calling the repository freeze-ready, complete all `Fix Now` items and verify:

1. `example10` has deterministic tests for both live-like and degraded paths.
2. No fallback market or macro data can drive a deployable directional stance without an explicit completeness rule.
3. Reports expose enough provenance for a reviewer to audit the output.
4. The documented setup and trust model match actual runtime behavior.
