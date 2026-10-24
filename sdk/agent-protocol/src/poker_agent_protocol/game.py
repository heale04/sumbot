"""What a bot sees and returns. Plain dataclasses: cheap to build, easy to read.

The engine builds these from trusted state, so they are not re-validated. Only a bot's
reply crosses a trust boundary; `action_from_dict` checks it strictly.
"""

from dataclasses import dataclass, field
from typing import Any

MAX_CHIPS = 2**53 - 1  # Exact in both Python and JavaScript JSON numbers.


@dataclass(frozen=True, slots=True)
class Action:
    """`kind` is fold, check, call, raise, show, muck, or a variant-defined kind such as discard.

    For raise, `amount` is the total this bot will have bet on the current betting round.
    `cards` are indexes into the bot's own cards, for variants that discard or choose cards.
    """

    kind: str
    amount: int = 0
    cards: tuple[int, ...] = ()

    @staticmethod
    def fold() -> "Action":
        return Action("fold")

    @staticmethod
    def check() -> "Action":
        return Action("check")

    @staticmethod
    def call() -> "Action":
        return Action("call")

    @staticmethod
    def raise_to(amount: int) -> "Action":
        return Action("raise", amount)

    @staticmethod
    def show() -> "Action":
        return Action("show")

    @staticmethod
    def muck() -> "Action":
        return Action("muck")


@dataclass(frozen=True, slots=True)
class LegalAction:
    """One allowed kind. For raise, amounts are the inclusive range of totals; for a
    variant-defined kind, the variant documents what the range means (e.g. card counts)."""

    kind: str
    min_amount: int = 0
    max_amount: int = 0


@dataclass(frozen=True, slots=True)
class BlindLevel:
    small_blind: int
    big_blind: int
    ante: int = 0


@dataclass(frozen=True, slots=True)
class TableConfig:
    """The table's fixed rules, the same on every turn at a table."""

    table_id: str
    seat: int
    players: int
    variant: str
    starting_stack: int
    blinds: tuple[BlindLevel, ...]  # The last level repeats.
    hands_per_level: int
    hand_cap: int
    payouts: tuple[int, ...]
    decision_ms: int
    memory_bytes: int  # Memory longer than this is cut to this length.
    time_bank_ms: int = 0  # Each bot's bank at the start of this table.


@dataclass(frozen=True, slots=True)
class Seat:
    seat: int  # Anonymized for this table.
    stack: int
    bet: int  # Chips bet on the current betting round.
    status: str  # active, folded, all_in, or busted.
    shown_cards: tuple[str, ...] = ()  # Face-up cards, e.g. at showdown.


@dataclass(frozen=True, slots=True)
class Event:
    """One public event: hand (a new hand starts; `amount` is its number), ante, blind, fold,
    check, call, raise, board, show, muck, win, and stack (a seat's stack when a hand ends)."""

    kind: str
    seat: int | None = None
    amount: int = 0
    cards: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GameState:
    # What happened at this table since this bot's last turn: the rest of earlier hands,
    # including showdowns and payouts, then this hand so far. Everything from the start of
    # the table on a bot's first turn there.
    events: tuple[Event, ...]
    # Hand
    hand_number: int
    seat: int
    cards: tuple[str, ...]  # This bot's private cards, e.g. ("As", "Td").
    board: tuple[str, ...]
    pot: int  # All chips committed this hand, including current bets.
    to_call: int
    legal_actions: tuple[LegalAction, ...]
    history: tuple[Event, ...]  # This hand so far.
    # Table
    seats: tuple[Seat, ...]
    button: int
    # Tournament
    blind_level: int
    blinds: BlindLevel
    hands_until_next_level: int
    players_remaining: int
    bust_order: tuple[int, ...]  # Seats, first to bust first.
    config: TableConfig
    variant: dict[str, Any] = field(default_factory=dict)  # Variant-specific public state.
    phase: str = "betting"  # Reference hold'em: betting or showdown; variants may add phases.
    # What is left of this bot's time bank at this table: a decision may take this much
    # longer than `config.decision_ms`.
    time_bank_ms: float = 0.0

    def can(self, kind: str) -> LegalAction | None:
        """The legal action of this kind, if any: `state.can("raise")`."""
        return next((legal for legal in self.legal_actions if legal.kind == kind), None)


def passive_action(legal_actions: tuple[LegalAction, ...]) -> Action:
    """What a bot does when it times out or errors: check if free, otherwise fold.

    In a variant-defined phase with neither, the first legal kind at its minimum amount.
    """
    kinds = {legal.kind for legal in legal_actions}
    if "check" in kinds:
        return Action.check()
    if "fold" in kinds:
        return Action.fold()
    return Action(legal_actions[0].kind, legal_actions[0].min_amount)


def is_legal(action: Action, legal_actions: tuple[LegalAction, ...]) -> bool:
    for legal in legal_actions:
        if legal.kind == action.kind:
            if action.kind == "raise":
                return legal.min_amount <= action.amount <= legal.max_amount
            if action.kind in ("show", "muck"):
                return action.amount == 0 and not action.cards
            return True
    return False


# Wire encoding. Hand-written because it runs on every decision.


def table_config_to_dict(config: TableConfig) -> dict[str, Any]:
    return {
        "table_id": config.table_id,
        "seat": config.seat,
        "players": config.players,
        "variant": config.variant,
        "starting_stack": config.starting_stack,
        "blinds": [[b.small_blind, b.big_blind, b.ante] for b in config.blinds],
        "hands_per_level": config.hands_per_level,
        "hand_cap": config.hand_cap,
        "payouts": list(config.payouts),
        "decision_ms": config.decision_ms,
        "memory_bytes": config.memory_bytes,
        "time_bank_ms": config.time_bank_ms,
    }


def table_config_from_dict(data: dict[str, Any]) -> TableConfig:
    return TableConfig(
        table_id=data["table_id"],
        seat=data["seat"],
        players=data["players"],
        variant=data["variant"],
        starting_stack=data["starting_stack"],
        blinds=tuple(BlindLevel(*level) for level in data["blinds"]),
        hands_per_level=data["hands_per_level"],
        hand_cap=data["hand_cap"],
        payouts=tuple(data["payouts"]),
        decision_ms=data["decision_ms"],
        memory_bytes=data["memory_bytes"],
        time_bank_ms=data.get("time_bank_ms", 0),
    )


def events_to_list(events: tuple[Event, ...]) -> list[list[Any]]:
    return [[e.kind, e.seat, e.amount, e.cards] for e in events]


def events_from_list(data: list[list[Any]]) -> tuple[Event, ...]:
    return tuple(Event(e[0], e[1], e[2], tuple(e[3])) for e in data)


def state_to_dict(state: GameState) -> dict[str, Any]:
    return {
        "events": events_to_list(state.events),
        "hand_number": state.hand_number,
        "seat": state.seat,
        "cards": state.cards,
        "board": state.board,
        "pot": state.pot,
        "to_call": state.to_call,
        "legal_actions": [[a.kind, a.min_amount, a.max_amount] for a in state.legal_actions],
        "history": events_to_list(state.history),
        "seats": [[s.seat, s.stack, s.bet, s.status, s.shown_cards] for s in state.seats],
        "button": state.button,
        "blind_level": state.blind_level,
        "blinds": [state.blinds.small_blind, state.blinds.big_blind, state.blinds.ante],
        "hands_until_next_level": state.hands_until_next_level,
        "players_remaining": state.players_remaining,
        "bust_order": state.bust_order,
        "config": table_config_to_dict(state.config),
        "variant": state.variant,
        "phase": state.phase,
        "time_bank_ms": state.time_bank_ms,
    }


def state_from_dict(data: dict[str, Any]) -> GameState:
    return GameState(
        events=events_from_list(data["events"]),
        hand_number=data["hand_number"],
        seat=data["seat"],
        cards=tuple(data["cards"]),
        board=tuple(data["board"]),
        pot=data["pot"],
        to_call=data["to_call"],
        legal_actions=tuple(LegalAction(*a) for a in data["legal_actions"]),
        history=events_from_list(data["history"]),
        seats=tuple(Seat(s[0], s[1], s[2], s[3], tuple(s[4])) for s in data["seats"]),
        button=data["button"],
        blind_level=data["blind_level"],
        blinds=BlindLevel(*data["blinds"]),
        hands_until_next_level=data["hands_until_next_level"],
        players_remaining=data["players_remaining"],
        bust_order=tuple(data["bust_order"]),
        config=table_config_from_dict(data["config"]),
        variant=data["variant"],
        phase=data.get("phase", "betting"),
        time_bank_ms=data.get("time_bank_ms", 0.0),
    )


def action_from_dict(data: object) -> Action:
    """Strictly parse an untrusted reply. Raises ValueError on anything unexpected."""
    if not isinstance(data, dict) or set(data) - {"kind", "amount", "cards"}:
        raise ValueError("An action is an object with kind, amount, and cards.")
    kind, amount, cards = data.get("kind"), data.get("amount", 0), data.get("cards", [])
    if not isinstance(kind, str) or not 0 < len(kind) <= 32:
        raise ValueError("Invalid action kind.")
    if type(amount) is not int or not 0 <= amount <= MAX_CHIPS:
        raise ValueError("Invalid action amount.")
    if (
        not isinstance(cards, list)
        or len(cards) > 64
        or any(type(card) is not int or not 0 <= card < 64 for card in cards)
    ):
        raise ValueError("Invalid action cards.")
    return Action(kind, amount, tuple(cards))
