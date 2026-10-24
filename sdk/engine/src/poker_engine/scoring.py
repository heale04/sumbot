"""Points for one finished table: place payouts for busted bots, an ICM split at the hand cap."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # The table logs equity each hand, so it imports this module.
    from poker_engine.table import TableResult


def icm(stacks: Sequence[int], payouts: Sequence[float]) -> list[float]:
    """Expected payout per player (Malmuth-Harville): each place is won in proportion to chips.

    Dynamic programming over the set of players already placed: O(2^n * n) for n <= 10.
    """
    players = [player for player in range(len(stacks)) if stacks[player] > 0]
    total = sum(stacks[player] for player in players)
    equity = [0.0] * len(stacks)
    places = min(len(payouts), len(players))
    # chance[mask]: probability that exactly the players in `mask` took the first places.
    chance = {0: 1.0}
    for place in range(places):
        following: dict[int, float] = {}
        for mask, probability in chance.items():
            remaining = total - sum(stacks[p] for i, p in enumerate(players) if mask >> i & 1)
            for i, player in enumerate(players):
                if mask >> i & 1:
                    continue
                share = probability * stacks[player] / remaining
                equity[player] += share * payouts[place]
                following[mask | 1 << i] = following.get(mask | 1 << i, 0.0) + share
        chance = following
    return equity


def equity(stacks: Sequence[int], bust_order: Sequence[int], payouts: Sequence[int]) -> list[float]:
    """Each seat's points if the table stopped now: busted seats keep their place's payout and
    the seats with chips split the rest by ICM."""
    seats = len(stacks)
    points = [0.0] * seats
    for index, seat in enumerate(bust_order):
        place = seats - 1 - index  # Zero-based: the first bust takes last place.
        points[seat] = float(payouts[place]) if place < len(payouts) else 0.0
    survivors = [seat for seat in range(seats) if stacks[seat] > 0]
    remaining = list(payouts[: len(survivors)])
    shares = icm([stacks[seat] for seat in survivors], remaining)
    for seat, value in zip(survivors, shares, strict=True):
        points[seat] = value
    return points


def table_points(result: "TableResult", payouts: Sequence[int]) -> list[float]:
    """Points per seat: the equity of the finished table."""
    return equity(result.final_stacks, result.bust_order, payouts)


def table_places(result: "TableResult") -> list[int]:
    """Finishing place per seat, 1 = first. Busted seats place by bust order; survivors at the
    hand cap rank by chips, ties by seat."""
    seats = len(result.final_stacks)
    places = [0] * seats
    for index, seat in enumerate(result.bust_order):
        places[seat] = seats - index
    survivors = sorted(
        (seat for seat in range(seats) if result.final_stacks[seat] > 0),
        key=lambda seat: (-result.final_stacks[seat], seat),
    )
    for place, seat in enumerate(survivors, start=1):
        places[seat] = place
    return places
