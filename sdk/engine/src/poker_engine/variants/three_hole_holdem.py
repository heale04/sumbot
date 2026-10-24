"""No-limit hold'em with three hole cards and no discard."""

import random

from poker_agent_protocol.game import BlindLevel

from poker_engine.holdem import HoldemHand


class ThreeHoleHoldem:
    name = "three_hole_holdem"
    title = "Three-hole hold'em"
    max_players = 10

    def deal(
        self, *, stacks: tuple[int, ...], blinds: BlindLevel, rng: random.Random
    ) -> HoldemHand:
        return HoldemHand(stacks=stacks, blinds=blinds, rng=rng, hole_cards=3)


VARIANT = ThreeHoleHoldem
