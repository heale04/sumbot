"""Play one table with one bot per seat. A failing bot plays passively; the table always finishes.

Bots are stateless: the table keeps each seat's memory and time bank, and hands them back on
that seat's next turn. Both start afresh at every table, so nothing carries from one table to
the next.
"""

from collections.abc import Sequence

from poker_agent_protocol.game import Action
from poker_agent_protocol.runner import Bot, BotError, BotTimeout

from poker_engine.table import SitAndGo, TableResult


async def play_table(table: SitAndGo, bots: Sequence[Bot]) -> TableResult:
    if len(bots) != table.rules.players:
        raise ValueError("Every seat needs exactly one bot.")
    failures = [0] * len(bots)
    decisions = [0] * len(bots)
    memory = [b""] * len(bots)
    while (state := table.decision()) is not None:
        seat = state.seat
        decisions[seat] += 1
        action: Action | None = None
        answered = False
        bank_ms = 0.0
        try:
            action, memory[seat], bank_ms = await bots[seat].act(state, memory[seat])
            answered = True
        except BotTimeout:
            bank_ms = state.time_bank_ms  # It waited out the whole bank.
        except BotError:
            pass  # The memory stays as it was, and these events are sent again next turn.
        if not table.apply(action, answered=answered, bank_ms=bank_ms):
            failures[seat] += 1
    return table.result(tuple(failures), tuple(decisions))
