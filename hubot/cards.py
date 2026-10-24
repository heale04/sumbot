"""Card encoding and a fast Numba hand evaluator for 5 to 8 cards.

Card = rank * 4 + suit, rank 0..12 (deuce..ace), suit 0..3 (c, d, h, s).
`evaluate(cards, n)` returns an integer: higher is a better best-five hand.
"""

import numpy as np
from numba import njit

RANKS = "23456789TJQKA"
SUITS = "cdhs"


def card_from_str(text: str) -> int:
    return RANKS.index(text[0]) * 4 + SUITS.index(text[1])


def card_to_str(card: int) -> str:
    return RANKS[card >> 2] + SUITS[card & 3]


def _build_tables():
    top5 = np.zeros(8192, dtype=np.int64)
    straight = np.full(8192, -1, dtype=np.int64)
    for mask in range(8192):
        bits = [r for r in range(12, -1, -1) if mask >> r & 1]
        value = 0
        for i in range(5):
            value = (value << 4) | (bits[i] + 1 if i < len(bits) else 0)
        top5[mask] = value
        best = -1
        for top in range(12, 3, -1):
            if all(mask >> (top - k) & 1 for k in range(5)):
                best = top
                break
        if best < 0 and (mask & 0b1000000001111) == 0b1000000001111:
            best = 3  # wheel: A2345, the five is high
        straight[mask] = best
    return top5, straight


TOP5, STRAIGHT = _build_tables()


@njit(cache=False, nogil=True)
def popcount13(x):
    c = 0
    while x:
        x &= x - 1
        c += 1
    return c


@njit(cache=False, nogil=True)
def evaluate(cards, n, top5, straight):
    """Best five-card hand value from the first n cards of `cards`."""
    suit_mask0 = 0
    suit_mask1 = 0
    suit_mask2 = 0
    suit_mask3 = 0
    rank_mask = 0
    cnt = np.zeros(13, dtype=np.int64)
    for i in range(n):
        c = cards[i]
        r = c >> 2
        s = c & 3
        if s == 0:
            suit_mask0 |= 1 << r
        elif s == 1:
            suit_mask1 |= 1 << r
        elif s == 2:
            suit_mask2 |= 1 << r
        else:
            suit_mask3 |= 1 << r
        rank_mask |= 1 << r
        cnt[r] += 1
    # straight flush first; a plain flush only after quads and full houses
    flush_value = -1
    for s in range(4):
        if s == 0:
            m = suit_mask0
        elif s == 1:
            m = suit_mask1
        elif s == 2:
            m = suit_mask2
        else:
            m = suit_mask3
        if popcount13(m) >= 5:
            sf = straight[m]
            if sf >= 0:
                return (8 << 20) | sf
            flush_value = (5 << 20) | top5[m]
    quads = -1
    trips = -1
    trips2 = -1
    pair1 = -1
    pair2 = -1
    for r in range(12, -1, -1):
        c = cnt[r]
        if c == 4:
            if quads < 0:
                quads = r
        elif c == 3:
            if trips < 0:
                trips = r
            elif trips2 < 0:
                trips2 = r
        elif c == 2:
            if pair1 < 0:
                pair1 = r
            elif pair2 < 0:
                pair2 = r
    if quads >= 0:
        rest = rank_mask & ~(1 << quads)
        return (7 << 20) | (quads << 4) | (top5[rest] >> 16)
    if trips >= 0 and (trips2 >= 0 or pair1 >= 0):
        low = trips2 if trips2 > pair1 else pair1
        return (6 << 20) | (trips << 4) | low
    if flush_value >= 0:
        return flush_value
    st = straight[rank_mask]
    if st >= 0:
        return (4 << 20) | st
    if trips >= 0:
        rest = rank_mask & ~(1 << trips)
        return (3 << 20) | (trips << 8) | (top5[rest] >> 12)
    if pair1 >= 0 and pair2 >= 0:
        rest = rank_mask & ~(1 << pair1) & ~(1 << pair2)
        return (2 << 20) | (pair1 << 8) | (pair2 << 4) | (top5[rest] >> 16)
    if pair1 >= 0:
        rest = rank_mask & ~(1 << pair1)
        return (1 << 20) | (pair1 << 12) | (top5[rest] >> 8)
    return top5[rank_mask]
