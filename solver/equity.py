"""Monte Carlo equity primitives shared by the abstraction builder and the runtime."""

import numpy as np
from numba import njit, prange

from solver.cards import STRAIGHT, TOP5, evaluate


@njit(cache=False, nogil=True)
def _remove(deck, n_deck, card):
    """Remove `card` from deck[:n_deck] by swapping with the last. Returns the new length."""
    for i in range(n_deck):
        if deck[i] == card:
            deck[i] = deck[n_deck - 1]
            deck[n_deck - 1] = card
            return n_deck - 1
    return n_deck


@njit(cache=False, nogil=True)
def equity_vs_random(hole, board, n_board, samples, top5, straight):
    """Equity (win + tie/2) of a three-card hand vs one uniformly random opponent hand,
    with the board completed uniformly at random, by Monte Carlo over `samples` runouts."""
    deck = np.empty(52, dtype=np.int64)
    for i in range(52):
        deck[i] = i
    n = 52
    for i in range(3):
        n = _remove(deck, n, hole[i])
    for i in range(n_board):
        n = _remove(deck, n, board[i])
    need = 3 + (5 - n_board)
    mine = np.empty(8, dtype=np.int64)
    theirs = np.empty(8, dtype=np.int64)
    for i in range(3):
        mine[i] = hole[i]
    for i in range(n_board):
        mine[3 + i] = board[i]
        theirs[3 + i] = board[i]
    score = 0.0
    for s in range(samples):
        # partial Fisher-Yates for `need` cards
        for k in range(need):
            j = k + np.random.randint(0, n - k)
            tmp = deck[k]
            deck[k] = deck[j]
            deck[j] = tmp
        theirs[0] = deck[0]
        theirs[1] = deck[1]
        theirs[2] = deck[2]
        for i in range(5 - n_board):
            mine[3 + n_board + i] = deck[3 + i]
            theirs[3 + n_board + i] = deck[3 + i]
        vm = evaluate(mine, 8, top5, straight)
        vt = evaluate(theirs, 8, top5, straight)
        if vm > vt:
            score += 1.0
        elif vm == vt:
            score += 0.5
    return score / samples


@njit(cache=False, nogil=True)
def showdown(hole0, hole1, board):
    """+1 if hole0 wins, -1 if it loses, 0 on a tie; board has five cards."""
    a = np.empty(8, dtype=np.int64)
    b = np.empty(8, dtype=np.int64)
    for i in range(3):
        a[i] = hole0[i]
        b[i] = hole1[i]
    for i in range(5):
        a[3 + i] = board[i]
        b[3 + i] = board[i]
    va = evaluate(a, 8, TOP5, STRAIGHT)
    vb = evaluate(b, 8, TOP5, STRAIGHT)
    if va > vb:
        return 1
    if va < vb:
        return -1
    return 0


@njit(cache=False, nogil=True)
def matchup_pair(h0_in, h1_in, samples, deck, h0, h1, board, perm):
    """P(win), P(tie) of class h0 vs class h1 (h1 under random suit permutations).
    Returns (-1, -1) after 5000 consecutive card collisions (an impossible pair)."""
    w = 0.0
    t = 0.0
    done = 0
    failures = 0
    while done < samples:
        if failures > 5000:
            return -1.0, -1.0
        for k in range(4):
            perm[k] = k
        for k in range(3, 0, -1):
            r = np.random.randint(0, k + 1)
            tmp = perm[k]
            perm[k] = perm[r]
            perm[r] = tmp
        for k in range(3):
            h0[k] = h0_in[k]
            c = h1_in[k]
            h1[k] = (c >> 2) * 4 + perm[c & 3]
        ok = True
        for k in range(3):
            for m in range(3):
                if h0[k] == h1[m]:
                    ok = False
        if not ok:
            failures += 1
            continue
        failures = 0
        m = 0
        for c in range(52):
            used = False
            for k in range(3):
                if h0[k] == c or h1[k] == c:
                    used = True
            if not used:
                deck[m] = c
                m += 1
        for k in range(5):
            r = k + np.random.randint(0, m - k)
            tmp = deck[k]
            deck[k] = deck[r]
            deck[r] = tmp
            board[k] = deck[k]
        res = showdown(h0, h1, board)
        if res > 0:
            w += 1.0
        elif res == 0:
            t += 1.0
        done += 1
    return w / samples, t / samples


@njit(parallel=True, cache=False)
def preflop_matchups(reps, samples, seed):
    """P(win) and P(tie) for every ordered pair of preflop classes, by Monte Carlo.

    The second class is dealt with a random suit permutation so the pair covers every
    suit relationship between the two hands. Impossible pairs count as a tie.
    """
    n = reps.shape[0]
    win = np.zeros((n, n), dtype=np.float32)
    tie = np.zeros((n, n), dtype=np.float32)
    for i in prange(n):
        np.random.seed(seed + i)
        deck = np.empty(52, dtype=np.int64)
        h0 = np.empty(3, dtype=np.int64)
        h1 = np.empty(3, dtype=np.int64)
        board = np.empty(5, dtype=np.int64)
        perm = np.empty(4, dtype=np.int64)
        for j in range(n):
            w, t = matchup_pair(reps[i], reps[j], samples, deck, h0, h1, board, perm)
            if w < 0.0:
                win[i, j] = 0.0
                tie[i, j] = 1.0
            else:
                win[i, j] = w
                tie[i, j] = t
    return win, tie
