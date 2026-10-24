"""Bot server inside a microVM: `python -m poker_agent_sdk.zygote BOT_DIR [--unix PATH]`.

Imports the bot once, then forks a copy for every host connection, so one VM per team
serves all of that team's tables on the machine without loading the bot (or torch) again.
Each copy speaks the same frames as `guest`. It listens on vsock port 1024, or on a Unix
socket for tests. Anything the bot prints goes to the VM's stderr, never to a connection.
"""

import contextlib
import os
import select
import signal
import socket
import sys

from poker_agent_sdk.guest import load, serve

PORT = 1024
HANGUP = select.POLLHUP | select.POLLERR | getattr(select, "POLLRDHUP", 0)


def listen(unix_path: str | None) -> socket.socket:
    if unix_path is not None:
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(unix_path)
    else:
        server = socket.socket(socket.AF_VSOCK, socket.SOCK_STREAM)  # type: ignore[attr-defined, unused-ignore]
        server.bind((socket.VMADDR_CID_ANY, PORT))  # type: ignore[attr-defined, unused-ignore]
    server.listen(64)
    return server


class Copy:
    """One forked copy of the bot, serving one host connection in its own process group."""

    def __init__(self, connection: socket.socket, pid: int) -> None:
        self.connection = connection
        self.pid = pid
        # Readable once the copy exits, so the host learns at once instead of by timeout.
        self.exited: int | None = None
        if hasattr(os, "pidfd_open"):
            try:
                self.exited = os.pidfd_open(pid)
            except ProcessLookupError:  # Already gone (and reaped): hang up now.
                connection.shutdown(socket.SHUT_RDWR)

    def stop(self) -> None:
        """Kill the copy and anything it started (its process group), then hang up."""
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(self.pid, signal.SIGKILL)  # In case it has not made its group yet.
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(self.pid, signal.SIGKILL)
        self.connection.close()
        if self.exited is not None:
            os.close(self.exited)


def main() -> None:
    args = sys.argv[1:]
    if len(args) not in (1, 3) or (len(args) == 3 and args[1] != "--unix"):
        sys.exit("usage: python -m poker_agent_sdk.zygote BOT_DIR [--unix PATH]")
    os.dup2(2, 1)  # The bot's prints never reach a connection.
    module = load(args[0])  # If it fails to load, every connection is told why.
    server = listen(args[2] if len(args) == 3 else None)
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)  # Finished copies are reaped automatically.
    poller = select.poll()
    poller.register(server, select.POLLIN)
    copies: dict[int, Copy] = {}  # By the file descriptor that reports on the copy.
    while True:
        for fd, events in poller.poll():
            if fd == server.fileno():
                try:
                    connection, _ = server.accept()
                except OSError:  # Out of file descriptors, say: PID 1 must never exit.
                    continue
                try:
                    pid = os.fork()
                except OSError:  # Out of memory or processes: refuse this one, keep serving.
                    connection.close()
                    continue
                if pid == 0:
                    os.setsid()  # Its own process group, so stopping it stops its children.
                    for other in {copy.connection for copy in copies.values()}:
                        other.close()
                    server.close()
                    signal.signal(signal.SIGCHLD, signal.SIG_DFL)
                    try:
                        serve(module, connection.makefile("rb"), connection.makefile("wb"))
                    finally:
                        os._exit(0)
                copy = Copy(connection, pid)
                copies[connection.fileno()] = copy
                poller.register(connection, HANGUP)  # Never POLLIN: the copy reads the data.
                if copy.exited is not None:
                    copies[copy.exited] = copy
                    poller.register(copy.exited, select.POLLIN)
            elif fd in copies and events & (HANGUP | select.POLLIN):
                # The host hung up (perhaps because the copy was too slow), or the copy died:
                # stop it, even mid-computation, and its children with it.
                copy = copies[fd]
                for watched in (copy.connection.fileno(), copy.exited):
                    if watched is not None:
                        poller.unregister(watched)
                        copies.pop(watched, None)
                copy.stop()


if __name__ == "__main__":
    main()
