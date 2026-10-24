"""Raises the minimum whenever it can, otherwise calls or checks. It remembers nothing."""

import poker_agent_sdk
from poker_agent_sdk import Action, GameState


class Agent(poker_agent_sdk.Agent):
    def act(self, state: GameState, memory: bytes) -> tuple[Action, bytes]:
        if state.can("show"):
            return Action.show(), memory
        if raise_ := state.can("raise"):
            return Action.raise_to(raise_.min_amount), memory
        return (Action.call() if state.can("call") else Action.check()), memory
