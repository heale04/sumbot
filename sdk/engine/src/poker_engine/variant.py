"""The interface a poker variant implements. The table handles everything between hands.

Players in a hand are numbered by position, with the button last. A variant only sees the
players who still have chips, and must take every random choice from the `rng` it is given.

Variants built into the engine are the modules of poker_engine.variants, named by ID, each
with its rules for teams beside it as <ID>.md. A variant may also live in a separate package
that registers itself, with its rules as a `rules` attribute:

    [project.entry-points."poker_engine.variants"]
    my_variant = "my_package:MyVariant"
"""

import pkgutil
import random
import re
from dataclasses import dataclass, field
from importlib import import_module, resources
from importlib.metadata import entry_points
from typing import Any, Protocol

from poker_agent_protocol.game import Action, BlindLevel, LegalAction


@dataclass(frozen=True, slots=True)
class HandView:
    """What one player may see of the hand right now."""

    cards: tuple[str, ...]  # The player's own private cards.
    board: tuple[str, ...]
    pot: int  # All chips committed this hand, including current bets.
    to_call: int
    stacks: tuple[int, ...]  # Per position.
    bets: tuple[int, ...]  # Per position, current betting round.
    folded: tuple[bool, ...]
    shown: tuple[tuple[str, ...], ...]  # Face-up cards per position.
    variant: dict[str, Any] = field(default_factory=dict)
    phase: str = "betting"


@dataclass(frozen=True, slots=True)
class HandEvent:
    kind: str
    player: int | None = None
    amount: int = 0
    cards: tuple[str, ...] = ()


class Hand(Protocol):
    def actor(self) -> int | None:
        """Deal as needed and return the position to act, or None once the hand is over."""
        ...

    def legal_actions(self) -> tuple[LegalAction, ...]:
        """Allowed actions. Without check/fold, the first action is the failure fallback."""
        ...

    def apply(self, action: Action) -> None:
        """Apply the actor's action. Raise ValueError, without changing state, if it is illegal."""
        ...

    def view(self, player: int) -> HandView: ...

    def events(self) -> tuple[HandEvent, ...]:
        """Every public event of this hand so far. Never include hidden cards."""
        ...

    def dealt(self) -> tuple[tuple[str, ...], ...]:
        """Every private card dealt to each position so far, for the hand log. Never shown to
        a bot, and kept even after the cards are mucked."""
        ...

    def stacks(self) -> tuple[int, ...]: ...


class Variant(Protocol):
    name: str  # The ID competitions and table rules refer to.
    title: str  # Shown to organizers and teams.
    max_players: int

    def deal(self, *, stacks: tuple[int, ...], blinds: BlindLevel, rng: random.Random) -> Hand: ...


# The public reference variant, and the table rules' default.
DEFAULT_VARIANT = "no_limit_holdem"


def installed_variants() -> list[str]:
    """Built-in variant IDs, then those registered by other packages."""
    from poker_engine import variants

    built_in = sorted(module.name for module in pkgutil.iter_modules(variants.__path__))
    others = sorted(entry.name for entry in entry_points(group="poker_engine.variants"))
    return built_in + [name for name in others if name not in built_in]


def load_variant(name: str) -> Variant:
    if re.fullmatch(r"[a-z0-9_]{1,64}", name):
        try:
            module = import_module(f"poker_engine.variants.{name}")
        except ModuleNotFoundError as error:
            if error.name != f"poker_engine.variants.{name}":
                raise
        else:
            if hasattr(module, "VARIANT"):
                built_in: Variant = module.VARIANT()
                return built_in
    for entry in entry_points(group="poker_engine.variants", name=name):
        variant: Variant = entry.load()()
        return variant
    raise KeyError(f"No installed poker variant is named {name!r}.")


def variant_rules(name: str) -> str:
    """Markdown for teams describing the variant."""
    rules = resources.files("poker_engine.variants").joinpath(f"{name}.md")
    if re.fullmatch(r"[a-z0-9_]{1,64}", name) and rules.is_file():
        return str(rules.read_text())
    return str(getattr(load_variant(name), "rules", ""))
