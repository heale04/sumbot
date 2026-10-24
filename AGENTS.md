# Instructions for AI coding assistants

You are helping a team write a poker bot for the HUQT Poker Bot Competition. Read this file and
`README.md` before writing code. These are the rules the platform enforces; code that breaks
them loses decisions or gets rejected.

## What the bot is

- One file, `agent.py`, defining `class Agent(poker_agent_sdk.Agent)` with
  `act(state, memory) -> (Action, bytes)`. It may read other files shipped next to it in the
  same zip.
- The Agent is stateless. The platform answers any table's turn with any copy of the bot, so
  nothing on `self` or in module globals may be relied on between calls (and using globals to
  carry information between tables breaks the rules). Everything the bot remembers at a table
  goes in `memory`: bytes returned with each action and handed back on its next turn at that
  table. It starts as `b""` at every table and is cut beyond `state.config.memory_bytes`
  (64 KB). It travels with every decision inside the 1 ms, so keep it under 16 KB and quick
  to (de)serialize, e.g. `json` or `struct`; a cut memory must not crash the bot, so parse
  it defensively.
- `act` is called on the bot's turn and must return within **1 ms** (the platform measures from
  its side, so aim well below that). Beyond that it spends the bot's time bank: 100 ms per
  table by default, full again at every table, with what is left in `state.time_bank_ms`; a
  timeout empties it. Too slow, an exception, or an illegal action means the
  bot checks if it can, otherwise folds, during betting. At reference hold'em showdown,
  failures automatically show if the hand can win against already shown cards, otherwise
  muck. A timeout or exception keeps the previous memory; an illegal parsed action still
  returns the bot's new memory.
- Legal actions come from `state.legal_actions` / `state.can("fold" | "check" | "call" |
"raise" | "show" | "muck")`. `Action.raise_to(amount)` takes the total bet for the betting round, within
  `[min_amount, max_amount]`. Never assume an action is legal; ask `state.can`.
- `state.phase` identifies `"betting"` or `"showdown"` in reference hold'em; other variants
  may define their own phases. Handle showdown in the same `act` call: `Action.show()` reveals
  both hole cards and `Action.muck()` surrenders the hand's claim, even if it would win.
  The last contender may only show. All-in reveals happen automatically without a bot turn.
  Opponents' shown cards appear in `state.seats[].shown_cards` and `show` events; a `muck`
  event contains no cards. Never return a betting action at showdown.
- `state.events` lists what happened at the table since the bot's last turn there (hand starts,
  actions, board cards, showdowns, payouts, final stacks); `state.history` covers the current
  hand; `state.config` has the table's fixed rules. Seat numbers are anonymized per table.

## The environment

- Python 3.12 with exactly the libraries in `requirements.txt` (NumPy, SciPy, pandas,
  scikit-learn, PyTorch CPU, LightGBM, XGBoost, ONNX Runtime, Numba, eval7) and the standard
  library. Nothing else can be imported. There is no GPU.
- No network access, no subprocesses or programs from the submission, and no writing files to
  keep state between tables. Bots must not run work in the background between decisions.
- Upload `agent.py` alone or a zip with `agent.py` at its root, up to 50 MB.
- `RULES.md` describes the variant this kit was made for, and the local simulator plays it.
  Write strategy code that reads the rules from `state.config`, the rest of `GameState`, and
  the legal actions, rather than hard-coding hold'em assumptions, where that is practical.

## How to help well

- Keep `act` simple and fast. Precompute tables at import time, avoid Python loops
  over large data in `act`, and measure: time `act` over many calls (`time.perf_counter_ns`).
- Test locally before every upload: `poker-agent check agent.py`, then
  `poker-agent play . examples/check_or_fold examples/min_raiser --tables 200`, and look at the
  failed decisions column. Zero failures is the goal.
- Use the SDK's real API (read `sdk/agent-sdk/src/poker_agent_sdk` and
  `sdk/agent-protocol/src/poker_agent_protocol/game.py`); do not invent methods.
- `poker_engine.scoring.icm(stacks, payouts)` computes ICM equity; payouts default to 50/30/20
  points for 1st/2nd/3rd, but a round can draw each table's own within ranges, so always use
  `state.config.payouts`. Surviving the hand cap splits the rest by ICM, so chip preservation
  matters near the money.
