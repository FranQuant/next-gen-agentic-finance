from datetime import datetime, timezone

import pytest

from core.pipeline.run_macro_slice import run_macro_slice


@pytest.mark.integration
def test_macro_slice_smoke() -> None:
    packet = run_macro_slice(
        "SPX500_USD",
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2026, 2, 17, tzinfo=timezone.utc),
    )

    assert set(packet.feature_values.keys()) == {
        "spx_20d_return",
        "spx_above_200dma",
        "dgs10_level",
        "dgs10_20d_change",
    }
    assert packet.regime is not None
    assert len(packet.rule_trace) >= 1
    assert packet.as_of is not None
