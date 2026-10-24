"""The contestant's base class. Subclassing it is optional; any `Agent` class with `act` works."""

from poker_agent_protocol.game import Action, GameState


class Agent:
    """Stateless: the platform may answer any table's turn with any copy of your bot, so keep
    nothing on `self` or in globals between calls. What you need to remember at a table goes
    in `memory`, bytes you return with each action and get back on your next turn there.
    Memory starts empty (b"") at every table, and is cut if longer than
    `state.config.memory_bytes` (64 KB by default). It travels with every decision, inside
    its time, so keep it small: up to about 16 KB costs nothing measurable.

    Load strategy files at module import time: they load once and every call shares them.
    """

    def act(self, state: GameState, memory: bytes) -> tuple[Action, bytes]:
        raise NotImplementedError
