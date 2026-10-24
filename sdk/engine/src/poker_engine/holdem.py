"""Reference variant: no-limit Texas hold'em through PokerKit."""

import random
from collections import deque
from functools import cache, cached_property

from poker_agent_protocol.game import Action, BlindLevel, LegalAction
from pokerkit import (
    AntePosting,
    Automation,
    BlindOrStraddlePosting,
    BoardDealing,
    CheckingOrCalling,
    ChipsPulling,
    CompletionBettingOrRaisingTo,
    Deck,
    Entry,
    Folding,
    HoleCardsShowingOrMucking,
    HoleDealing,
    NoLimitTexasHoldem,
    StandardHighHand,
)

from poker_engine.variant import HandEvent, HandView

AUTOMATIONS = (
    Automation.ANTE_POSTING,
    Automation.BET_COLLECTION,
    Automation.BLIND_OR_STRADDLE_POSTING,
    Automation.HAND_KILLING,
    Automation.CHIPS_PUSHING,
    Automation.CHIPS_PULLING,
)
DECK = tuple(sorted(Deck.STANDARD, key=repr))


class _CachedHighHand(StandardHighHand):
    # PokerKit compares each immutable hand repeatedly while finding and paying winners.
    # Keep its evaluator and ranking; avoid looking up the same hand's entry each time.
    @cached_property
    def entry(self) -> Entry:
        return super().entry


class _CachedHoldem(NoLimitTexasHoldem):
    hand_types = (_CachedHighHand,)


@cache
def _game(hole_cards: int) -> type[NoLimitTexasHoldem]:
    """Hold'em dealing this many hole cards. The hand is the best five of hole and board cards
    together, using any number of each."""
    if hole_cards == 2:
        return _CachedHoldem
    return type(f"_CachedHoldem{hole_cards}", (_CachedHoldem,), {"hole_dealing_count": hole_cards})


class NoLimitHoldem:
    name = "no_limit_holdem"
    title = "No-limit hold'em"
    max_players = 10

    def deal(
        self, *, stacks: tuple[int, ...], blinds: BlindLevel, rng: random.Random
    ) -> "HoldemHand":
        return HoldemHand(stacks=stacks, blinds=blinds, rng=rng)


class HoldemHand:
    def __init__(
        self,
        *,
        stacks: tuple[int, ...],
        blinds: BlindLevel,
        rng: random.Random,
        hole_cards: int = 2,
    ) -> None:
        self._total = sum(stacks)
        self._state = _game(hole_cards).create_state(
            AUTOMATIONS,
            True,
            blinds.ante,
            (blinds.small_blind, blinds.big_blind),
            blinds.big_blind,
            stacks,
            len(stacks),
        )
        # PokerKit shuffles with the global RNG; replace its deck before anything is dealt.
        deck = list(DECK)
        rng.shuffle(deck)
        self._state.deck_cards = deque(deck)
        self._events: list[HandEvent] = []
        self._shown: list[tuple[str, ...]] = [()] * len(stacks)
        self._dealt: list[tuple[str, ...]] = [()] * len(stacks)
        self._recorded = 0
        self._settled = False

    def actor(self) -> int | None:
        state = self._state
        while True:
            if state.can_deal_hole():
                state.deal_hole()
            elif state.can_burn_card():
                state.burn_card()
            elif state.can_deal_board():
                state.deal_board()
            elif state.showdown_index is not None:
                if state.all_in_status:
                    state.show_or_muck_hole_cards(True)
                else:
                    return state.showdown_index
            else:
                return state.actor_index

    def legal_actions(self) -> tuple[LegalAction, ...]:
        state = self._state
        if state.showdown_index is not None:
            player = state.showdown_index
            assert player is not None
            if sum(state.statuses) == 1:
                return (LegalAction("show"),)  # Every pot needs an eligible contender.
            # No check/fold at showdown: the first legal kind is the failure fallback.
            # This uses only this player's hand and opponents' already shown cards.
            automatic = "show" if state.can_win_now(player) else "muck"
            return (LegalAction(automatic), LegalAction("muck" if automatic == "show" else "show"))
        actor = state.actor_index
        assert actor is not None
        legal = []
        if state.can_fold():
            legal.append(LegalAction("fold"))
        if state.can_check_or_call():
            legal.append(LegalAction("check" if max(state.bets) == state.bets[actor] else "call"))
        if state.can_complete_bet_or_raise_to():
            low = state.min_completion_betting_or_raising_to_amount
            high = state.max_completion_betting_or_raising_to_amount
            assert low is not None and high is not None
            legal.append(LegalAction("raise", min(low, high), high))
        return tuple(legal)

    def apply(self, action: Action) -> None:
        state = self._state
        if action.kind in ("show", "muck"):
            if (
                action.amount
                or action.cards
                or action.kind not in {legal.kind for legal in self.legal_actions()}
            ):
                raise ValueError("That showdown action is not legal.")
            state.show_or_muck_hole_cards(action.kind == "show")
        elif action.kind == "fold":
            state.fold()
        elif action.kind in ("check", "call"):
            state.check_or_call()
        elif action.kind == "raise":
            state.complete_bet_or_raise_to(action.amount)
        else:
            raise ValueError(f"No-limit hold'em has no {action.kind} action.")

    def view(self, player: int) -> HandView:
        state = self._state
        self.events()
        return HandView(
            cards=tuple(repr(card) for card in state.hole_cards[player]),
            board=tuple(repr(dealt[0]) for dealt in state.board_cards),
            pot=self._total - sum(state.stacks),
            to_call=max(state.bets) - state.bets[player],
            stacks=tuple(state.stacks),
            bets=tuple(state.bets),
            folded=tuple(not status for status in state.statuses),
            shown=tuple(self._shown),
            phase="showdown" if state.showdown_index is not None else "betting",
        )

    def events(self) -> tuple[HandEvent, ...]:
        """Convert new public PokerKit operations. Deals and burns stay private."""
        operations = self._state.operations
        for operation in operations[self._recorded :]:
            match operation:
                case HoleDealing(player_index=player, cards=cards):
                    self._dealt[player] += tuple(map(repr, cards))
                case AntePosting(player_index=player, amount=amount):
                    self._events.append(HandEvent("ante", player, amount))
                case BlindOrStraddlePosting(player_index=player, amount=amount):
                    self._events.append(HandEvent("blind", player, amount))
                case BoardDealing(cards=cards):
                    self._events.append(HandEvent("board", cards=tuple(map(repr, cards))))
                case Folding(player_index=player):
                    self._events.append(HandEvent("fold", player))
                case CheckingOrCalling(player_index=player, amount=amount):
                    self._events.append(HandEvent("call" if amount else "check", player, amount))
                case CompletionBettingOrRaisingTo(player_index=player, amount=amount):
                    self._events.append(HandEvent("raise", player, amount))
                case HoleCardsShowingOrMucking(player_index=player, hole_cards=cards) if cards:
                    self._shown[player] = tuple(map(repr, cards))
                    self._events.append(HandEvent("show", player, cards=self._shown[player]))
                case HoleCardsShowingOrMucking(player_index=player):
                    self._events.append(HandEvent("muck", player))
                case ChipsPulling(player_index=player, amount=amount) if amount:
                    self._events.append(HandEvent("win", player, amount))
        self._recorded = len(operations)
        if not self._state.status and not self._settled:
            # Uncalled bets are refunded without an operation, so state final stacks outright.
            self._settled = True
            for player, stack in enumerate(self._state.stacks):
                self._events.append(HandEvent("stack", player, stack))
        return tuple(self._events)

    def dealt(self) -> tuple[tuple[str, ...], ...]:
        self.events()
        return tuple(self._dealt)

    def stacks(self) -> tuple[int, ...]:
        return tuple(self._state.stacks)
