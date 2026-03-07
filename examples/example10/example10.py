from __future__ import annotations

import os
import sys
from typing import Any

try:
    from .team import build_example10_system
except ImportError:  # pragma: no cover
    from team import build_example10_system

try:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    _HAS_RICH = True
except Exception:  # pragma: no cover
    _HAS_RICH = False


def _render_rich(packet: Any) -> None:
    console = Console()

    title = Panel.fit(
        "[bold cyan]Example10[/bold cyan]\nMulti-Source Research Orchestrator",
        border_style="cyan",
    )
    console.print(title)

    brief_table = Table(title="Research Brief", box=box.SIMPLE_HEAD, show_header=True)
    brief_table.add_column("Field", style="bold")
    brief_table.add_column("Value")
    brief_table.add_row("Query", packet.brief.query)
    brief_table.add_row("Objective", packet.brief.objective)
    brief_table.add_row("Universe", ", ".join(packet.brief.tickers) or "N/A")
    brief_table.add_row("Topics", ", ".join(packet.brief.topics[:4]) or "N/A")
    brief_table.add_row("Macro Lens", ", ".join(packet.brief.macro_indicators) or "N/A")
    brief_table.add_row("Horizon", packet.brief.timeframe)
    console.print(brief_table)

    evidence_table = Table(title="Evidence", box=box.SIMPLE_HEAD, show_header=True)
    evidence_table.add_column("#", justify="right", width=3)
    evidence_table.add_column("Source", style="magenta", width=22)
    evidence_table.add_column("Headline / Summary")
    evidence_items = packet.evidence[:6]
    if evidence_items:
        for idx, item in enumerate(evidence_items, start=1):
            snippet = item.title or item.summary or "N/A"
            evidence_table.add_row(str(idx), item.source, snippet[:120])
    else:
        evidence_table.add_row("-", "N/A", "No external evidence available.")
    console.print(evidence_table)

    interpretation_lines = [
        f"Web: {packet.web_view.get('summary', 'N/A')}",
        f"Market: {packet.market_view.get('summary', 'N/A')}",
        f"Macro: {packet.macro_view.get('summary', 'N/A')}",
    ]
    macro_snapshot = packet.macro_view.get("indicator_snapshot", {})
    if isinstance(macro_snapshot, dict) and macro_snapshot:
        indicator_text = ", ".join(
            f"{k}={v:.2f}%"
            for k, v in macro_snapshot.items()
            if isinstance(v, (int, float))
        )
        if indicator_text:
            interpretation_lines.append(f"Macro Indicators: {indicator_text}")

    console.print(
        Panel(
            "\n".join(f"- {line}" for line in interpretation_lines),
            title="Interpretation",
            border_style="blue",
        )
    )

    allocation_text = ", ".join(
        f"{ticker}:{weight:.2f}" for ticker, weight in packet.portfolio_view.allocations.items()
    )
    portfolio_lines = [
        f"Signal: {packet.portfolio_view.signal}",
        f"Conviction: {packet.portfolio_view.conviction:.2f}",
        f"Horizon: {packet.portfolio_view.horizon}",
        f"Allocations: {allocation_text or 'N/A'}",
    ]
    if packet.portfolio_view.thesis:
        portfolio_lines.append("Thesis: " + " | ".join(packet.portfolio_view.thesis[:2]))

    console.print(
        Panel(
            "\n".join(f"- {line}" for line in portfolio_lines),
            title="Portfolio Implication",
            border_style="green",
        )
    )

    risks = packet.portfolio_view.risks[:5] or ["No explicit risks were produced."]
    console.print(
        Panel(
            "\n".join(f"- {risk}" for risk in risks),
            title="Risks",
            border_style="red",
        )
    )

    console.print(
        Panel.fit(
            f"[bold]RUN_ID[/bold]: {packet.run_id}",
            border_style="white",
        )
    )


def main() -> None:
    query = " ".join(sys.argv[1:]).strip()
    if not query:
        query = "Analyze MSFT and NVDA for a 3-12 month portfolio view."

    system = build_example10_system()
    packet = system.run(query)

    force_plain = os.getenv("EXAMPLE10_PLAIN_REPORT", "").strip() == "1"
    if _HAS_RICH and not force_plain:
        _render_rich(packet)
    else:
        print(packet.report)
        print()
        print(f"RUN_ID: {packet.run_id}")


if __name__ == "__main__":
    main()
