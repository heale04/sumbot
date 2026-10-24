"""Minimal bot: check when free, otherwise fold. It remembers nothing."""

import poker_agent_sdk
from poker_agent_sdk import Action, GameState


class Agent(poker_agent_sdk.Agent):
    def act(self, state: GameState, memory: bytes) -> tuple[Action, bytes]:
        if state.can("show"):
            return Action.show(), memory
        return (Action.check() if state.can("check") else Action.fold()), memory
