"""Individual leg agent reports for the boss."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LegReport:
    """One specialist agent's opinion on the watchlist."""

    agent_id: str
    scores: dict[str, float] = field(default_factory=dict)
    available: bool = True
    top_ticker: str | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.scores and not self.top_ticker:
            self.top_ticker = max(self.scores.items(), key=lambda x: x[1])[0]


def rank_normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    if len(scores) == 1:
        return {next(iter(scores)): 1.0}
    ranked = sorted(scores.items(), key=lambda x: x[1])
    n = len(ranked)
    return {ticker: i / (n - 1) for i, (ticker, _) in enumerate(ranked)}
