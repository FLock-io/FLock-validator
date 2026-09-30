from __future__ import annotations

import json
import math
import os
import select
import struct
import time
from typing import Any, BinaryIO

import numpy as np

# Wire format: an 8-byte big-endian length, then that many bytes of UTF-8 JSON
# holding one object. Only JSON ever crosses the boundary: the host never
# unpickles or otherwise interprets bytes produced by untrusted worker code.
HEADER = struct.Struct("!Q")

# Requests are small metadata (file paths, scalars); bulk data such as video
# frames travels through files, never through the pipe. Responses carry a
# trainer-produced result, so they get a larger, still bounded, allowance.
MAX_REQUEST_BYTES = 1024**2
MAX_RESPONSE_BYTES = 4 * 1024**2

_MAX_JSON_DEPTH = 64


class MessageTooLargeError(ValueError):
    """An outgoing message exceeds the size cap for its direction."""


def encode_message(message: dict[str, Any], *, max_bytes: int) -> bytes:
    payload = json.dumps(message, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    if len(payload) > max_bytes:
        raise MessageTooLargeError(
            f"message of {len(payload)} bytes exceeds the {max_bytes} byte limit"
        )
    return HEADER.pack(len(payload)) + payload


def write_message(stream: BinaryIO, message: dict[str, Any], *, max_bytes: int) -> None:
    data = encode_message(message, max_bytes=max_bytes)
    stream.write(data)
    stream.flush()


def read_exact(fd: int, size: int, deadline: float) -> bytes:
    """Read exactly ``size`` bytes from ``fd`` before the monotonic ``deadline``.

    Raises ``TimeoutError`` past the deadline and ``EOFError`` if the peer closes
    the pipe first.
    """
    chunks = bytearray()
    while len(chunks) < size:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError
        readable, _, _ = select.select([fd], [], [], remaining)
        if not readable:
            raise TimeoutError
        chunk = os.read(fd, size - len(chunks))
        if not chunk:
            raise EOFError("sandbox closed the protocol pipe")
        chunks.extend(chunk)
    return bytes(chunks)


def read_message(fd: int, deadline: float, *, max_bytes: int) -> dict[str, Any]:
    """Host side: read one framed JSON object with a wall-clock deadline."""
    header = read_exact(fd, HEADER.size, deadline)
    size = HEADER.unpack(header)[0]
    if size > max_bytes:
        raise MessageTooLargeError(
            f"peer announced a {size} byte message; the limit is {max_bytes}"
        )
    return _decode(read_exact(fd, size, deadline))


def read_blocking(stream: BinaryIO, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            raise EOFError("protocol pipe closed by peer")
        chunks.extend(chunk)
    return bytes(chunks)


def read_message_blocking(stream: BinaryIO, *, max_bytes: int) -> dict[str, Any]:
    """Worker side: block until the host sends one framed JSON object."""
    header = read_blocking(stream, HEADER.size)
    size = HEADER.unpack(header)[0]
    if size > max_bytes:
        raise MessageTooLargeError(
            f"peer announced a {size} byte message; the limit is {max_bytes}"
        )
    return _decode(read_blocking(stream, size))


def _decode(payload: bytes) -> dict[str, Any]:
    message = json.loads(payload.decode("utf-8"))
    if not isinstance(message, dict):
        raise ValueError("message must be a JSON object")
    return message


def to_jsonable(value: Any, _depth: int = 0) -> Any:
    """Convert untrusted worker output into plain JSON types.

    Accepts numpy scalars/arrays, tuples, and nested string-keyed dict/list.
    Rejects NaN/inf (not valid JSON) and every other type with ``TypeError``, so
    nothing surprising is ever serialised across the trust boundary.
    """
    if _depth > _MAX_JSON_DEPTH:
        raise TypeError("value is nested too deeply to serialise")
    depth = _depth + 1
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, np.generic):
        # np.float64 is also a float subclass; normalise via .item() first so
        # the checks below see a plain Python scalar.
        return to_jsonable(value.item(), depth)
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist(), depth)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        number = float(value)
        if not math.isfinite(number):
            raise TypeError("value contains a NaN or infinite float")
        return number
    if isinstance(value, str):
        return str.__str__(value)
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("dictionary keys must be strings")
            result[str.__str__(key)] = to_jsonable(item, depth)
        return result
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item, depth) for item in value]
    raise TypeError(f"value of type {type(value).__name__} is not JSON-serialisable")
