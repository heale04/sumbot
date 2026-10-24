"""Submissions: a single `agent.py`, or a zip with `agent.py` at its root plus any files it needs.

Checking never imports or runs the code. It bounds sizes, rejects unsafe paths and links,
and requires a top-level `Agent` class. Running the bot is still the real test.
"""

import ast
import hashlib
import io
import os
import stat
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZIP_STORED, BadZipFile, ZipFile, ZipInfo


class SubmissionError(ValueError):
    pass


@dataclass(frozen=True)
class Limits:
    max_upload_bytes: int = 50 * 1024 * 1024
    max_expanded_bytes: int = 200 * 1024 * 1024
    max_files: int = 1000
    max_source_bytes: int = 1024 * 1024  # agent.py is parsed in memory.
    max_path_length: int = 512
    max_path_depth: int = 16


DEFAULT_LIMITS = Limits()


def quoted(path: str) -> str:
    """A path from the upload, made safe to show back to its uploader: ASCII only, short."""
    return ascii(path[:80])


@dataclass(frozen=True)
class Report:
    sha256: str
    size_bytes: int
    files: tuple[str, ...]


def check_agent_source(data: bytes, *, limits: Limits = DEFAULT_LIMITS) -> None:
    if len(data) > limits.max_source_bytes:
        raise SubmissionError("agent.py is too large; put data in separate files.")
    try:
        tree = ast.parse(data.decode("utf-8-sig"), filename="agent.py")  # Python allows a BOM.
    except (UnicodeError, SyntaxError, ValueError, RecursionError, MemoryError):
        raise SubmissionError("agent.py is not valid UTF-8 Python.") from None
    if not any(isinstance(node, ast.ClassDef) and node.name == "Agent" for node in tree.body):
        raise SubmissionError("agent.py must define a top-level Agent class.")


def check_submission(data: bytes, *, limits: Limits = DEFAULT_LIMITS) -> Report:
    """Check uploaded bytes: a zip archive, or otherwise the source of agent.py itself."""
    if not data or len(data) > limits.max_upload_bytes:
        raise SubmissionError("The submission is empty or too large.")
    files = _check_zip(data, limits) if data[:4] == b"PK\x03\x04" else _check_source(data, limits)
    return Report(hashlib.sha256(data).hexdigest(), len(data), files)


def _check_source(data: bytes, limits: Limits) -> tuple[str, ...]:
    check_agent_source(data, limits=limits)
    return ("agent.py",)


def _check_directory_size(data: bytes, limits: Limits) -> None:
    """Bound the central directory from its end record before zipfile parses all of it.

    zipfile reads the directory by its byte size, not its declared entry count, so both
    are checked: a spoofed count alone would not stop a huge directory.
    """
    end = data.rfind(b"PK\x05\x06", max(0, len(data) - 65_557))
    if end < 0 or end + 22 > len(data):
        raise SubmissionError("The archive is not a valid zip file.")
    count, size = struct.unpack_from("<HI", data, end + 10)
    per_entry = 46 + limits.max_path_length + 128  # Fixed header, name, and typical extras.
    if count > limits.max_files or size > limits.max_files * per_entry:
        raise SubmissionError("The archive has too many files.")  # 0xFFFF also means zip64.


def _check_zip(data: bytes, limits: Limits) -> tuple[str, ...]:
    files: list[str] = []
    seen: set[str] = set()  # Case-folded paths of every entry.
    files_folded: set[str] = set()  # Case-folded paths of file entries.
    expanded = 0
    _check_directory_size(data, limits)
    try:
        with ZipFile(io.BytesIO(data)) as archive:
            entries = archive.infolist()
            if len(entries) > limits.max_files:
                raise SubmissionError("The archive has too many files.")
            for entry in entries:
                name = entry.filename.removesuffix("/")
                parts = name.split("/")
                # 255 bytes is the most a file name may have on common filesystems.
                if (
                    len(name) > limits.max_path_length
                    or len(parts) > limits.max_path_depth
                    or any(len(part.encode()) > 255 for part in parts)
                ):
                    raise SubmissionError(f"Archive path too long or deep: {quoted(name)}.")
                path = PurePosixPath(name)
                # The checked path must be exactly the path extraction will write.
                if (
                    entry.orig_filename != entry.filename
                    or "\\" in entry.filename
                    or not name
                    or path.as_posix() != name
                    or path.is_absolute()
                    or any(part in (".", "..") for part in parts)
                ):
                    raise SubmissionError(f"Unsafe path in archive: {quoted(entry.filename)}.")
                folded = name.casefold()
                if folded in seen:
                    raise SubmissionError(f"Duplicate path in archive: {quoted(entry.filename)}.")
                # Nothing may live under a file, and a file may not have anything under it.
                prefixes = ["/".join(parts[:depth]).casefold() for depth in range(1, len(parts))]
                if any(prefix in files_folded for prefix in prefixes) or (
                    not entry.is_dir() and any(other.startswith(folded + "/") for other in seen)
                ):
                    raise SubmissionError(f"A file and a directory share a path: {quoted(name)}.")
                seen.add(folded)
                if not entry.is_dir():
                    files_folded.add(folded)
                kind = stat.S_IFMT(entry.external_attr >> 16)
                if kind not in (0, stat.S_IFDIR if entry.is_dir() else stat.S_IFREG):
                    raise SubmissionError("Archives may contain only files and directories.")
                if entry.flag_bits & 1 or entry.compress_type not in (ZIP_STORED, ZIP_DEFLATED):
                    raise SubmissionError(
                        "Encrypted or unusually compressed archives are rejected."
                    )
                if entry.is_dir():
                    continue
                expanded += entry.file_size
                if expanded > limits.max_expanded_bytes:
                    raise SubmissionError("The archive expands beyond the size limit.")
                # Read everything so CRCs are verified and declared sizes cannot lie.
                with archive.open(entry) as member:
                    read = 0
                    chunks = []
                    while chunk := member.read(1024 * 1024):
                        read += len(chunk)
                        if read > entry.file_size:
                            raise SubmissionError("An archive file is larger than declared.")
                        if entry.filename == "agent.py":
                            if read > limits.max_source_bytes:
                                raise SubmissionError("agent.py is too large.")
                            chunks.append(chunk)
                if entry.filename == "agent.py":
                    check_agent_source(b"".join(chunks), limits=limits)
                files.append(entry.filename)
    except SubmissionError:
        raise
    except (
        BadZipFile,
        ValueError,
        OSError,
        RuntimeError,
        NotImplementedError,
        zlib.error,
        EOFError,
    ):
        # Corrupt offsets and names surface as all of these; every one is a bad upload.
        raise SubmissionError("The archive is not a valid zip file.") from None
    if "agent.py" not in files:
        raise SubmissionError("The archive needs agent.py at its root.")
    return tuple(sorted(files))


def pack(source: Path, destination: Path, *, limits: Limits = DEFAULT_LIMITS) -> Report:
    """Zip a bot directory reproducibly, skipping hidden files and __pycache__."""
    if not (source / "agent.py").is_file():
        raise SubmissionError("The directory needs an agent.py.")
    if destination.exists():
        raise SubmissionError("The destination already exists.")
    paths = []
    for directory, subdirectories, filenames in os.walk(source):
        subdirectories[:] = sorted(
            name for name in subdirectories if not name.startswith(".") and name != "__pycache__"
        )
        for name in filenames:
            path = Path(directory) / name
            if name.startswith("."):
                continue
            if path.is_symlink():
                raise SubmissionError(f"Links are not allowed: {path}.")
            paths.append(path)
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            entry = ZipInfo(path.relative_to(source).as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
            entry.external_attr = (stat.S_IFREG | 0o644) << 16
            entry.compress_type = ZIP_DEFLATED
            archive.writestr(entry, path.read_bytes())
    data = buffer.getvalue()
    report = check_submission(data, limits=limits)
    with destination.open("xb") as output:
        output.write(data)
    return report
