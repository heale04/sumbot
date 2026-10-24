"""Frames between the host and a bot: a JSON message plus raw bytes (the bot's memory).

A frame is two big-endian 32-bit lengths, the JSON, then the bytes. Memory travels as raw
bytes because it can be large; JSON would inflate and slow it.

Host to bot:  {"type": "act", "id": n, "state": {...}} with the seat's memory
Bot to host:  {"type": "ready"} once, after importing the bot
              {"type": "action", "id": n, "action": {...}, "truncated": bool} with new memory
              {"type": "error", "id": n, "detail": "..."} when act raised
"""

import json
import struct
from typing import Any, BinaryIO

MAX_MESSAGE_BYTES = 1024 * 1024
MAX_MEMORY_BYTES = 16 * 1024 * 1024  # A ceiling; each table's rules set the real limit.
HEADER = struct.Struct(">II")


class ProtocolError(ValueError):
    pass


def encode(message: dict[str, Any], memory: bytes = b"") -> bytes:
    data = json.dumps(message, separators=(",", ":")).encode()
    if len(data) > MAX_MESSAGE_BYTES or len(memory) > MAX_MEMORY_BYTES:
        raise ProtocolError("The message is too large.")
    return HEADER.pack(len(data), len(memory)) + data + memory


def decode(data: bytes) -> dict[str, Any]:
    try:
        message = json.loads(data)
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise ProtocolError("The message is not valid JSON.") from None
    if not isinstance(message, dict):
        raise ProtocolError("A message is a JSON object.")
    return message


def frame_lengths(header: bytes) -> tuple[int, int]:
    length, memory = HEADER.unpack(header)
    if not 0 < length <= MAX_MESSAGE_BYTES or memory > MAX_MEMORY_BYTES:
        raise ProtocolError("The message is empty or too large.")
    return int(length), int(memory)


def read_frame(source: BinaryIO) -> tuple[dict[str, Any], bytes] | None:
    """Blocking read for the bot side. None means the host closed the stream."""
    header = source.read(HEADER.size)
    if not header:
        return None
    if len(header) != HEADER.size:
        raise ProtocolError("The stream ended inside a frame.")
    length, memory = frame_lengths(header)
    data = source.read(length + memory)
    if len(data) != length + memory:
        raise ProtocolError("The stream ended inside a frame.")
    return decode(data[:length]), data[length:]
