"""Synthetic states that exercise every decision path once at import time."""

from poker_agent_sdk import BlindLevel, Event, GameState, LegalAction, Seat, TableConfig


def states():
    cfg = TableConfig("warm", 0, 8, "three_hole_holdem", 2000, (BlindLevel(10, 20),), 25, 200, (50, 30, 20), 1, 65536, 100)

    def st(cards, board, history, seats, to_call, pot, legal, seat=0, button=1):
        return GameState(events=(), hand_number=1, seat=seat, cards=cards, board=board, pot=pot, to_call=to_call,
                         legal_actions=legal, history=history, seats=seats, button=button, blind_level=1,
                         blinds=BlindLevel(10, 20), hands_until_next_level=25, players_remaining=len(seats),
                         bust_order=(), config=cfg)

    legal = (LegalAction("fold"), LegalAction("call"), LegalAction("raise", 40, 2000))
    legal_chk = (LegalAction("check"), LegalAction("raise", 20, 1900))
    out = []
    # heads-up preflop (we are SB) and river facing a bet
    h = (Event("blind", 0, 10), Event("blind", 1, 20))
    hu = (Seat(0, 1990, 10, "active"), Seat(1, 1980, 20, "active")) + tuple(Seat(i, 0, 0, "busted") for i in range(2, 8))
    out.append(st(("As", "Kd", "7c"), (), h, hu, 10, 30, legal))
    h2 = h + (Event("raise", 0, 60), Event("call", 1, 40), Event("board", cards=("2c", "9d", "Th")), Event("check", 1),
              Event("check", 0), Event("board", cards=("Qs",)), Event("check", 1), Event("check", 0),
              Event("board", cards=("3h",)), Event("raise", 1, 100))
    hu2 = (Seat(0, 1940, 0, "active"), Seat(1, 1840, 100, "active")) + tuple(Seat(i, 0, 0, "busted") for i in range(2, 8))
    out.append(st(("As", "Kd", "7c"), ("2c", "9d", "Th", "Qs", "3h"), h2, hu2, 100, 220,
                  (LegalAction("fold"), LegalAction("call"), LegalAction("raise", 200, 1940))))
    # 8-player preflop: UTG+1 facing an open, and the big blind with everyone folded
    seats8 = tuple(Seat(i, 2000, 0, "active") for i in range(8))
    h8 = (Event("blind", 2, 10), Event("blind", 3, 20), Event("raise", 4, 50))
    out.append(st(("Qh", "Jh", "9h"), (), h8, seats8, 50, 80, legal, seat=5, button=1))
    h8b = (Event("blind", 2, 10), Event("blind", 3, 20)) + tuple(Event("fold", i) for i in (4, 5, 6, 7, 0, 1)) + (Event("call", 2, 10),)
    out.append(st(("Qh", "Jh", "9h"), (), h8b, seats8, 0, 40, legal_chk, seat=3, button=1))
    # 3-way flop after a raise and two calls: we are the big blind, first to act
    h3 = (Event("blind", 2, 10), Event("blind", 3, 20), Event("raise", 4, 50)) + tuple(Event("fold", i) for i in (5, 6, 7, 0, 1)) + \
         (Event("call", 2, 40), Event("call", 3, 30), Event("board", cards=("2c", "9d", "Th")))
    seats3 = tuple(Seat(i, 1950 if i in (2, 3, 4) else 2000, 0, "active" if i in (2, 3, 4) else "folded") for i in range(8))
    out.append(st(("Qh", "Jh", "9h"), ("2c", "9d", "Th"), h3, seats3, 0, 150, legal_chk, seat=3, button=1))
    # same hand on the turn facing a bet
    h4 = h3 + (Event("check", 3), Event("raise", 4, 100), Event("fold", 2), Event("call", 3, 100), Event("board", cards=("Qs",)),
               Event("check", 3), Event("raise", 4, 200))
    seats4 = tuple(Seat(i, 1850 if i == 3 else 1750 if i == 4 else 1950 if i == 2 else 2000, 200 if i == 4 else 0,
                        "active" if i in (3, 4) else "folded") for i in range(8))
    out.append(st(("Qh", "Jh", "9h"), ("2c", "9d", "Th", "Qs"), h4, seats4, 200, 550,
                  (LegalAction("fold"), LegalAction("call"), LegalAction("raise", 400, 1850)), seat=3, button=1))
    return out
