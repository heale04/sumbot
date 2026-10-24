"""Bot process entry point: `python -m poker_agent_sdk.guest BOT_DIR`.

Imports `BOT_DIR/agent.py` once, makes one `Agent`, and answers `act` requests from any
table: the bot is stateless, and each request carries the memory the bot returned on its last
turn at that table. Memory longer than the table's limit is cut to that length. Frames go to
the original stdout; anything the contestant prints goes to stderr so it cannot corrupt the
protocol. Errors in `act` are logged and reported, and the host plays the passive action.
"""

import contextlib
import importlib
import math
import operator
import os
import sys
import traceback
from pathlib import Path
from typing import Any, BinaryIO

from poker_agent_protocol.game import Action, state_from_dict
from poker_agent_protocol.process import ERROR_CHARS
from poker_agent_protocol.wire import encode, read_frame


def chips(value: Any) -> int:
    """Accept integers (including NumPy's) and finite floats rounded down, as bot math
    produces. Booleans and strings are rejected rather than silently converted."""
    if isinstance(value, bool):
        raise TypeError("A chip amount cannot be a boolean.")
    with contextlib.suppress(TypeError):
        return operator.index(value)
    if not hasattr(type(value), "__float__"):
        raise TypeError(f"A chip amount must be a number, not {type(value).__name__}.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("A chip amount must be finite.")
    return math.floor(number)


def reply_action(action: Any) -> dict[str, Any]:
    if not isinstance(action, Action):
        raise TypeError(f"act must return an Action, not {type(action).__name__}.")
    return {
        "kind": action.kind,
        "amount": chips(action.amount),
        "cards": [operator.index(card) for card in action.cards],
    }


def reply(result: Any) -> tuple[dict[str, Any], bytes]:
    """`act`'s `(action, memory)` as the reply's action and memory bytes."""
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError("act must return a pair: (action, memory).")
    action, memory = result
    if not isinstance(memory, bytes | bytearray | memoryview):
        raise TypeError(f"Memory must be bytes, not {type(memory).__name__}.")
    return reply_action(action), bytes(memory)


def load(bot_dir: str) -> Any:
    """Import the bot's agent.py once. If that fails, return the exception, to report it."""
    sys.path.insert(0, str(Path(bot_dir).resolve()))
    try:
        return importlib.import_module("agent")
    except BaseException as error:  # Including SystemExit from the bot's own code.
        traceback.print_exc()
        return error


def report() -> str:
    """The current exception for the host: the end of the traceback, which says the most."""
    traceback.print_exc()
    return traceback.format_exc()[-ERROR_CHARS:]


def serve(module: Any, source: BinaryIO, output: BinaryIO) -> None:
    """Answer one host connection until it closes. If the bot failed to load (`module` is the
    exception) or its Agent cannot be made, the host is told why, then the connection ends."""

    def send(message: dict[str, Any], memory: bytes = b"") -> None:
        output.write(encode(message, memory))
        output.flush()

    try:
        if isinstance(module, BaseException):
            raise module
        agent = module.Agent()
    except BaseException as error:  # Including SystemExit from the bot's own code.
        detail = "".join(traceback.format_exception(error))[-ERROR_CHARS:]
        send({"type": "error", "id": 0, "detail": detail})
        return
    send({"type": "ready"})
    while (frame := read_frame(source)) is not None:
        message, memory = frame
        try:
            state = state_from_dict(message["state"])
            action, memory = reply(agent.act(state, memory))
            limit = state.config.memory_bytes
            send(
                {
                    "type": "action",
                    "id": message["id"],
                    "action": action,
                    "truncated": len(memory) > limit,
                },
                memory[:limit],
            )
        except Exception:
            send({"type": "error", "id": message["id"], "detail": report()})


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: python -m poker_agent_sdk.guest BOT_DIR")
    output = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    serve(load(sys.argv[1]), sys.stdin.buffer, output)


if __name__ == "__main__":
    main()
