# HUQT Poker Bot Competition: starter kit

Everything you need to build, test, and submit a bot. The full guide, with the rules and the
schedule, is on the competition site under **Getting started**.

## 1. Set up (Python 3.12)

On Linux, macOS 14 or newer, or Windows (WSL works best):

```sh
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ./sdk/agent-protocol -e ./sdk/engine -e ./sdk/agent-sdk
```

`requirements.txt` holds the exact libraries the competition runs; nothing else is available
to your bot. The SDK gives you the `Agent` base class, a local tournament simulator, and the
upload checker.

The SDK packages are installed in editable mode: changing their Python source under `sdk/`
takes effect on the next run without reinstalling. Reinstall after changing package
dependencies. Local engine changes do not change the competition server; use the released
rules when testing a submission.

## 2. Write your bot

Edit `agent.py`. `act(state, memory)` runs on your turn and returns your action and the memory
you want back on your next turn at that table: bytes, empty at the start of every table, up to
64 KB. It must return within 1 ms; beyond that it spends your time bank, 100 ms per table and
full again at every table (`state.time_bank_ms` is what is left). Memory travels within your
1 ms, so keep it under 16 KB.
Your bot keeps nothing else between calls.
`examples/` has two simple bots to play against.

## 3. Test it locally

```sh
poker-agent check agent.py
poker-agent play . examples/check_or_fold examples/min_raiser --tables 200
```

`play` runs real sit-and-go tables between bot folders (repeat a folder to give it more seats)
and prints each bot's average points, finishing places, and failed decisions. Aim for zero
failed decisions: a betting failure checks or folds for you; a showdown failure uses the
automatic show/muck behavior described below.

## 4. Submit

Upload `agent.py` on your team page, or zip a folder (with `agent.py` at its root) using
`poker-agent pack . my_bot.zip`. Every upload plays a short test match first; when it shows
**Ready**, choose it as your team's active bot.

## Table rules (defaults)

- 8 players, 2,000 chips each. Blinds rise every 25 hands: 10/20, 15/30, 25/50, 40/80,
  60/120, 100/200, 150/300, 250/500 (the last level repeats).
- A table ends when one bot has every chip, or after 200 hands. Places pay 50/30/20 points;
  at the hand cap the remaining bots split the rest by ICM (`poker_engine.scoring.icm`).
- Two bots busting on the same hand: the one that started it with more chips places higher.

## Showing or mucking (reference hold'em)

`act` also receives showdown turns, with `state.phase == "showdown"`. Choose `Action.show()`
to reveal both cards or `Action.muck()` when `state.can("muck")` permits it. A muck surrenders
your claim to the pot, even with the best hand. The supplied template and examples always
show when asked. Showdown decisions use the same time limit and memory as betting.

Later players see cards already shown in `state.seats[].shown_cards` and `show` events.
A `muck` event names the seat without revealing cards. Folded cards stay hidden. All-in
showdowns reveal all remaining hands automatically. The last contender must show if everyone
else mucks, so the pot always has an eligible winner.

A timeout, error, or illegal showdown action shows if your hand can win against already shown
cards and mucks otherwise. Returning `fold`, `check`, `call`, or `raise` during showdown is
illegal and counts as a failed decision. Check the offered legal actions on every turn.
Other variants define their own phases and showdown rules.

## Modifying the kit

- `agent.py`: your strategy and table memory.
- `examples/`: simple reference opponents.
- `sdk/agent-protocol/src/poker_agent_protocol/game.py`: state, actions, and wire encoding.
- `sdk/engine/src/poker_engine/variant.py`: the hand/variant interface.
- `RULES.md`: the rules of your competition's variant.
- `sdk/engine/src/poker_engine/variants/`: that variant's implementation.
- `sdk/engine/src/poker_engine/holdem.py`: the hold'em reference most variants build on.
- `sdk/engine/src/poker_engine/table.py`: blinds, stacks, payouts, and hand limits.
- `sdk/agent-sdk/src/poker_agent_sdk/cli.py`: local simulation and submission commands.

Repeat a local match with `--seed YOUR_SEED` to compare changes with the same initial
randomness. Identical results require the same rules and bot actions as well as the seed.
Local play uses the variant this kit was made for; `RULES.md` describes it. Strategies
written for one variant's cards and legal actions may need revising for another.

## Using an AI assistant

`AGENTS.md` (and `CLAUDE.md`, which points to it) tells AI coding assistants the rules and
limits in this kit, so open this folder in your assistant and it will pick them up.
