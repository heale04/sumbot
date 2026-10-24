"""Runtime Monte Carlo equity (copy of the solver's estimator so buckets match)."""

import numpy as np
from numba import njit

from hubot.cards import STRAIGHT, TOP5, evaluate


@njit(cache=False, nogil=True)
def _remove(deck, n_deck, card):
    for i in range(n_deck):
        if deck[i] == card:
            deck[i] = deck[n_deck - 1]
            deck[n_deck - 1] = card
            return n_deck - 1
    return n_deck


@njit(cache=False, nogil=True)
def equity_vs_random(hole, board, n_board, samples, top5, straight):
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
def seed_rng(seed):
    np.random.seed(seed)


def equity(hole_cards, board_cards, samples):
    hole = np.asarray(hole_cards, dtype=np.int64)
    board = np.zeros(5, dtype=np.int64)
    nb = len(board_cards)
    if nb:
        board[:nb] = board_cards
    return equity_vs_random(hole, board, nb, samples, TOP5, STRAIGHT)
