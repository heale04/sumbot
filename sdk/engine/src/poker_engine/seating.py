"""Seat a tournament field for one iteration. The same seed and iteration always give the same tables."""

import random
from collections.abc import Hashable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SeatAssignment[Entry: Hashable]:
    entry: Entry
    filler: bool  # Fills a short table; plays normally but never scores.


def seat_iteration[Entry: Hashable](
    entries: Sequence[Entry], *, seed: str, iteration: int, table_size: int = 8
) -> list[list[SeatAssignment[Entry]]]:
    """Every entry plays exactly one table. A short last table is filled with entries from
    other tables, chosen at random; seat order within each table is random too."""
    if len(set(entries)) != len(entries) or len(entries) < 2:
        raise ValueError("A field needs at least two distinct entries.")
    rng = random.Random(f"{seed}:{iteration}")
    order = list(entries)
    rng.shuffle(order)
    tables = [
        [SeatAssignment(entry, False) for entry in order[start : start + table_size]]
        for start in range(0, len(order), table_size)
    ]
    last = tables[-1]
    missing = table_size - len(last)
    if missing:
        present = {seat.entry for seat in last}
        pool = [entry for entry in entries if entry not in present]
        last += [SeatAssignment(entry, True) for entry in rng.sample(pool, min(missing, len(pool)))]
        rng.shuffle(last)
    return tables
