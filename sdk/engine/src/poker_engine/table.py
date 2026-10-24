"""One sit-and-go: seats, button, blind levels, bust order, and the hand cap.

The variant plays each hand; this module handles everything between hands, builds the
`GameState` each bot sees, and logs every hand. Every random choice comes from the table seed.
"""

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from poker_agent_protocol.game import (
    MAX_CHIPS,
    Action,
    BlindLevel,
    Event,
    GameState,
    LegalAction,
    Seat,
    TableConfig,
    is_legal,
    passive_action,
)
from poker_agent_protocol.wire import MAX_MEMORY_BYTES

from poker_engine.scoring import equity
from poker_engine.variant import DEFAULT_VARIANT, Hand, HandEvent, Variant

DEFAULT_BLINDS = tuple(
    BlindLevel(small, big)
    for small, big in [
        (10, 20),
        (15, 30),
        (25, 50),
        (40, 80),
        (60, 120),
        (100, 200),
        (150, 300),
        (250, 500),
    ]
)

# The lowest and highest points one place can pay; equal for a fixed payout.
Payout = tuple[int, int]


@dataclass(frozen=True, slots=True)
class TableRules:
    variant: str = DEFAULT_VARIANT
    players: int = 8
    starting_stack: int = 2000
    blinds: tuple[BlindLevel, ...] = DEFAULT_BLINDS  # The last level repeats.
    hands_per_level: int = 25
    hand_cap: int = 200
    payouts: tuple[Payout, ...] = ((50, 50), (30, 30), (20, 20))  # Each table draws its own.
    decision_ms: int = 1
    # Per bot, full again at every table: time a decision may take beyond `decision_ms`.
    time_bank_ms: int = 100
    # Per bot per table; longer memory is cut. It travels with each decision, inside its time.
    memory_bytes: int = 64 * 1024

    def __post_init__(self) -> None:
        if not 2 <= self.players <= 10:
            raise ValueError("A table needs 2 to 10 players.")
        if not 1 <= self.starting_stack <= MAX_CHIPS // self.players:
            raise ValueError("The starting stack must be positive and fit exact chip counts.")
        if not 1 <= len(self.blinds) <= 100 or any(
            not 1 <= level.small_blind <= level.big_blind or level.ante < 0 for level in self.blinds
        ):
            raise ValueError("Each blind level needs 1 <= small blind <= big blind and ante >= 0.")
        if not (
            1 <= self.hands_per_level <= 10_000
            and 1 <= self.hand_cap <= 10_000
            and 1 <= self.decision_ms <= 60_000
            and 0 <= self.time_bank_ms <= 60_000
            and 0 <= self.memory_bytes <= MAX_MEMORY_BYTES
        ):
            raise ValueError("Hand counts, the decision time, or memory are outside their limits.")
        if any(
            type(low) is not int or type(high) is not int or not 0 <= low <= high <= MAX_CHIPS
            for low, high in self.payouts
        ):
            raise ValueError("Each payout must be a whole number, or a range low to high, >= 0.")


@dataclass(frozen=True, slots=True)
class TableResult:
    hands_played: int
    final_stacks: tuple[int, ...]  # Per seat.
    bust_order: tuple[int, ...]  # Seats, first to bust first.
    failures: tuple[int, ...]  # Per seat: timeouts, errors, and illegal actions.
    decisions: tuple[int, ...] = ()  # Per seat: decisions asked for.


class SitAndGo:
    def __init__(self, *, rules: TableRules, variant: Variant, table_id: str, seed: str) -> None:
        if rules.players > variant.max_players:
            raise ValueError(f"{variant.name} supports at most {variant.max_players} players.")
        self.rules = rules
        self.variant = variant
        self.table_id = table_id
        self.stacks = [rules.starting_stack] * rules.players
        self.bust_order: list[int] = []
        self.hands_played = 0
        self._seed = seed
        self.payouts = table_payouts(rules.payouts, seed)
        self._rng = random.Random(f"{seed}:table")
        self._button = self._rng.randrange(rules.players)
        self._hand: Hand | None = None
        self._positions: list[int] = []  # Hand position -> seat; the button is last.
        self._legal: tuple[LegalAction, ...] = ()  # Empty unless a decision is pending.
        # Public events of finished hands, and how many of all events each seat has seen.
        self._past: list[Event] = []
        self._seen = [0] * rules.players
        self._delivering = (0, 0)  # The pending decision's seat and its events' end.
        self._bank = [float(rules.time_bank_ms)] * rules.players  # Milliseconds left per seat.
        # One record per finished hand, for the hand log. Holds every card, so never for bots.
        self.log: list[dict[str, Any]] = []
        self._equity = self._current_equity()

    def config(self, seat: int) -> TableConfig:
        rules = self.rules
        return TableConfig(
            table_id=self.table_id,
            seat=seat,
            players=rules.players,
            variant=rules.variant,
            starting_stack=rules.starting_stack,
            blinds=rules.blinds,
            hands_per_level=rules.hands_per_level,
            hand_cap=rules.hand_cap,
            payouts=self.payouts,
            decision_ms=rules.decision_ms,
            memory_bytes=rules.memory_bytes,
            time_bank_ms=rules.time_bank_ms,
        )

    def decision(self) -> GameState | None:
        """The next decision, or None when the table is finished."""
        while True:
            if self._hand is None:
                if sum(stack > 0 for stack in self.stacks) < 2:
                    return None
                if self.hands_played >= self.rules.hand_cap:
                    return None
                self._start_hand()
            assert self._hand is not None
            player = self._hand.actor()
            if player is not None:
                self._legal = self._hand.legal_actions()
                return self._state(player)
            self._end_hand()

    def apply(self, action: Action | None, *, answered: bool = True, bank_ms: float = 0.0) -> bool:
        """Apply the bot's action, or the passive action if it is missing or illegal.

        `answered` is false when the bot never replied (a timeout or crash): its memory did
        not change, so the events it was sent are sent again on its next turn. `bank_ms` is
        the time this decision took from the seat's time bank.
        """
        legal, self._legal = self._legal, ()
        seat, seen = self._delivering
        self._bank[seat] = max(0.0, self._bank[seat] - bank_ms)
        if answered:
            self._seen[seat] = seen
        if self._hand is None or not legal:
            raise RuntimeError("No decision is pending.")
        if action is not None and is_legal(action, legal):
            try:
                self._hand.apply(action)
                return True
            except Exception:  # noqa: BLE001 - A bot-shaped action must never fail the job.
                pass
        self._hand.apply(passive_action(legal))
        return False

    def result(self, failures: tuple[int, ...], decisions: tuple[int, ...] = ()) -> TableResult:
        return TableResult(
            hands_played=self.hands_played,
            final_stacks=tuple(self.stacks),
            bust_order=tuple(self.bust_order),
            failures=failures,
            decisions=decisions,
        )

    def _level(self) -> int:
        return min(self.hands_played // self.rules.hands_per_level, len(self.rules.blinds) - 1)

    def _start_hand(self) -> None:
        players = self.rules.players
        if self.hands_played:
            self._button = next(
                seat
                for offset in range(1, players + 1)
                if self.stacks[seat := (self._button + offset) % players] > 0
            )
        self._positions = [
            seat
            for offset in range(1, players + 1)
            if self.stacks[seat := (self._button + offset) % players] > 0
        ]
        self._hand = self.variant.deal(
            stacks=tuple(self.stacks[seat] for seat in self._positions),
            blinds=self.rules.blinds[self._level()],
            # Its own stream per hand: cards seen in earlier hands say nothing about this deck.
            rng=random.Random(f"{self._seed}:hand:{self.hands_played}"),
        )

    def _end_hand(self) -> None:
        assert self._hand is not None
        before = list(self.stacks)
        for position, stack in enumerate(self._hand.stacks()):
            self.stacks[self._positions[position]] = stack
        if sum(self.stacks) != sum(before):
            raise RuntimeError("Chips were not conserved; the variant is broken.")
        busted = [seat for seat in self._positions if self.stacks[seat] == 0]
        # Fewer chips at the start of the hand busts first; exact ties are decided by the seed.
        self._rng.shuffle(busted)
        self.bust_order += sorted(busted, key=lambda seat: before[seat])
        self._past += self._events()
        self._record(before)
        self.hands_played += 1
        self._hand = None

    def _record(self, before: list[int]) -> None:
        assert self._hand is not None
        level = self.rules.blinds[self._level()]
        cards: list[list[str]] = [[] for _ in self.stacks]
        for position, dealt in enumerate(self._hand.dealt()):
            cards[self._positions[position]] = list(dealt)
        previous, self._equity = self._equity, self._current_equity()
        self.log.append(
            {
                "hand": self.hands_played + 1,
                "button": self._button,
                "blinds": [level.small_blind, level.big_blind, level.ante],
                "stacks_before": before,
                "stacks_after": list(self.stacks),
                "cards": cards,
                "events": [logged(e, self._positions) for e in self._hand.events()],
                "equity_before": previous,
                "equity_after": self._equity,
            }
        )

    def _current_equity(self) -> list[float]:
        """Points per seat if the table stopped now, rounded for the log."""
        points = equity(self.stacks, self.bust_order, self.payouts)
        return [round(value, 4) for value in points]

    def _events(self) -> tuple[Event, ...]:
        """This hand's public events, with hand positions translated to seats, after an
        event marking the hand's start."""
        assert self._hand is not None
        positions = self._positions
        return (Event("hand", amount=self.hands_played + 1),) + tuple(
            Event(e.kind, None if e.player is None else positions[e.player], e.amount, e.cards)
            for e in self._hand.events()
        )

    def _unseen(self, seat: int, current: tuple[Event, ...]) -> tuple[Event, ...]:
        """Every event since this seat's last answered decision."""
        past, seen = self._past, self._seen[seat]
        self._delivering = (seat, len(past) + len(current))
        return tuple(past[seen:]) + current[max(0, seen - len(past)) :]

    def _state(self, player: int) -> GameState:
        assert self._hand is not None
        view = self._hand.view(player)
        positions = self._positions
        seats = []
        in_hand = {seat: position for position, seat in enumerate(positions)}
        for seat in range(self.rules.players):
            position = in_hand.get(seat)
            if position is None:
                seats.append(Seat(seat, 0, 0, "busted"))
                continue
            stack = view.stacks[position]
            status = "folded" if view.folded[position] else "all_in" if stack == 0 else "active"
            seats.append(Seat(seat, stack, view.bets[position], status, view.shown[position]))
        level = self._level()
        history = self._events()
        return GameState(
            events=self._unseen(positions[player], history),
            hand_number=self.hands_played + 1,
            seat=positions[player],
            cards=view.cards,
            board=view.board,
            pot=view.pot,
            to_call=view.to_call,
            legal_actions=self._legal,
            history=history[1:],
            seats=tuple(seats),
            button=self._button,
            blind_level=level + 1,
            blinds=self.rules.blinds[level],
            hands_until_next_level=self.rules.hands_per_level
            - self.hands_played % self.rules.hands_per_level,
            players_remaining=len(positions),
            bust_order=tuple(self.bust_order),
            config=self.config(positions[player]),
            variant=view.variant,
            phase=view.phase,
            time_bank_ms=self._bank[positions[player]],
        )


def table_seed(seed: str, iteration: int, number: int) -> str:
    """The seed of one table of a tournament: its cards, button, ties, and payouts."""
    return f"{seed}:{iteration}:{number}"


def table_payouts(payouts: Sequence[Payout], seed: str) -> tuple[int, ...]:
    """One table's points per place: a whole number within each place's range, drawn from the
    table's seed, then sorted so a higher place never pays less."""
    rng = random.Random(f"{seed}:payouts")
    return tuple(sorted((rng.randint(low, high) for low, high in payouts), reverse=True))


def logged(event: HandEvent, positions: list[int]) -> dict[str, Any]:
    """One event for the hand log, by seat, without empty fields."""
    entry: dict[str, Any] = {"kind": event.kind}
    if event.player is not None:
        entry["seat"] = positions[event.player]
    if event.amount:
        entry["amount"] = event.amount
    if event.cards:
        entry["cards"] = list(event.cards)
    return entry


def rules_to_dict(rules: TableRules) -> dict[str, object]:
    """JSON-ready rules, for freezing a tournament's configuration."""
    return {
        "variant": rules.variant,
        "players": rules.players,
        "starting_stack": rules.starting_stack,
        "blinds": [[b.small_blind, b.big_blind, b.ante] for b in rules.blinds],
        "hands_per_level": rules.hands_per_level,
        "hand_cap": rules.hand_cap,
        "payouts": [low if low == high else [low, high] for low, high in rules.payouts],
        "decision_ms": rules.decision_ms,
        "time_bank_ms": rules.time_bank_ms,
        "memory_bytes": rules.memory_bytes,
    }


def rules_from_dict(data: dict[str, Any]) -> TableRules:
    """Inverse of rules_to_dict. Missing keys take the spec defaults; unknown keys fail."""
    values = dict(data)
    if "blinds" in values:
        values["blinds"] = tuple(BlindLevel(*level) for level in values["blinds"])
    if "payouts" in values:
        values["payouts"] = tuple(
            (value, value) if isinstance(value, int) else tuple(value)
            for value in values["payouts"]
        )
    return TableRules(**values)
