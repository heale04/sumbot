# Three-hole hold'em

No-limit hold'em with three hole cards instead of two. Nothing is discarded: you keep all
three cards for the whole hand.

- Blinds, betting rounds, and the flop, turn and river work as in no-limit hold'em.
- At showdown your hand is the best five of your three hole cards and the five board cards,
  using any number of hole cards, from none to all three.
- Players who reach showdown without being all in choose to show or muck; mucking gives up
  the pot. `state.cards` holds your three cards.
