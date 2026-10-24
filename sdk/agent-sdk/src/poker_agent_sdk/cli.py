"""`poker-agent pack`, `check`, and `play`. Only `play` runs bot code, as ordinary local processes."""

import argparse
import asyncio
import json
import secrets
import sys
from collections import Counter
from pathlib import Path

from poker_agent_protocol.process import BotPool
from poker_engine.play import play_table
from poker_engine.scoring import table_places, table_points
from poker_engine.seating import seat_iteration
from poker_engine.table import SitAndGo, TableRules
from poker_engine.variant import DEFAULT_VARIANT, installed_variants, load_variant

from poker_agent_sdk.bundle import SubmissionError, check_submission, pack


def main() -> None:
    parser = argparse.ArgumentParser(prog="poker-agent")
    commands = parser.add_subparsers(dest="command", required=True)
    pack_command = commands.add_parser("pack", help="Zip a bot directory for submission.")
    pack_command.add_argument("source", type=Path)
    pack_command.add_argument("destination", type=Path)
    check_command = commands.add_parser("check", help="Check a submission without running it.")
    check_command.add_argument("submission", type=Path)
    play = commands.add_parser("play", help="Play local bot directories against each other.")
    play.add_argument(
        "bots", type=Path, nargs="+", help="Directories with agent.py; repeat to add seats."
    )
    play.add_argument("--tables", type=int, default=20)
    play.add_argument("--seed", default=None)
    installed = installed_variants()
    # A starter kit installs only its competition's variant, so that is the default there.
    play.add_argument(
        "--variant",
        default=installed[0] if len(installed) == 1 else DEFAULT_VARIANT,
        help=f"Installed: {', '.join(installed)}",
    )
    play.add_argument("--decision-ms", type=int, default=1)
    play.add_argument("--hand-cap", type=int, default=200)
    arguments = parser.parse_args()
    try:
        if arguments.command == "pack":
            report = pack(arguments.source, arguments.destination)
        elif arguments.command == "check":
            report = check_submission(arguments.submission.read_bytes())
        else:
            asyncio.run(play_local(arguments))
            return
    except (SubmissionError, OSError, ValueError, KeyError) as error:
        parser.exit(2, f"{error}\n")
    print(
        json.dumps(
            {"sha256": report.sha256, "size_bytes": report.size_bytes, "files": report.files},
            indent=2,
        )
    )


async def play_local(arguments: argparse.Namespace) -> None:
    bots: list[Path] = [bot.resolve() for bot in arguments.bots]
    if not 2 <= len(bots) <= 10 or not all((bot / "agent.py").is_file() for bot in bots):
        raise ValueError("Give 2 to 10 bot directories that each contain agent.py.")
    seed = arguments.seed or secrets.token_hex(8)
    rules = TableRules(
        variant=arguments.variant,
        players=len(bots),
        decision_ms=arguments.decision_ms,
        hand_cap=arguments.hand_cap,
    )
    variant = load_variant(rules.variant)
    # Two copies per directory, shared by every seat that repeats it, as in production.
    processes = {
        bot: BotPool(
            [sys.executable, "-m", "poker_agent_sdk.guest", str(bot)], copies=2, cwd=str(bot)
        )
        for bot in set(bots)
    }
    running = asyncio.Semaphore(8)  # One event loop measures every deadline; keep it responsive.
    labels = [f"{bot.name}#{index}" for index, bot in enumerate(bots)]
    points: dict[str, list[float]] = {label: [] for label in labels}
    places: dict[str, Counter[int]] = {label: Counter() for label in labels}
    failures: Counter[str] = Counter()

    async def one_table(number: int) -> None:
        seating = seat_iteration(labels, seed=seed, iteration=number, table_size=len(bots))[0]
        table = SitAndGo(
            rules=rules, variant=variant, table_id=f"local-{number}", seed=f"{seed}:{number}"
        )
        # The table keeps each seat's memory, so a directory used twice plays two seats.
        seats = [processes[bots[labels.index(seat.entry)]] for seat in seating]
        async with running:
            result = await play_table(table, seats)
        finishes = zip(table_points(result, table.payouts), table_places(result), strict=True)
        for seat, (assignment, (value, place)) in enumerate(zip(seating, finishes, strict=True)):
            points[assignment.entry].append(value)
            failures[assignment.entry] += result.failures[seat]
            places[assignment.entry][place] += 1

    try:
        await asyncio.gather(*(one_table(number) for number in range(arguments.tables)))
    finally:
        for process in processes.values():
            await process.stop()
    for bot, process in processes.items():
        if process.logs:
            print(f"--- {bot.name} stderr (last {len(process.logs)} bytes) ---", file=sys.stderr)
            print(process.logs.decode(errors="replace"), file=sys.stderr)
    print(
        json.dumps(
            {
                "seed": seed,
                "tables": arguments.tables,
                "bots": {
                    label: {
                        "average_points": round(sum(points[label]) / len(points[label]), 3),
                        "places": dict(sorted(places[label].items())),
                        "failed_decisions": failures[label],
                    }
                    for label in labels
                },
            },
            indent=2,
        )
    )
