"""Card abstraction: per-street equity buckets, Markov transitions, and showdown matrices.

The abstract game treats each player's cards as an independent Markov chain of buckets:
preflop class (exact suit-isomorphism class), then flop, turn, and river buckets defined by
quantiles of equity vs a random hand. Showdown payoffs between buckets on each street are
estimated on shared boards, which keeps the board correlation where it matters most.
"""

import time

import numpy as np
from numba import njit, prange

from solver.cards import STRAIGHT, TOP5
from solver.equity import equity_vs_random, preflop_matchups, showdown
from solver.iso import CLASS_OF, N_CLASSES, REPRESENTATIVE, WEIGHT


@njit(parallel=True, cache=False)
def sample_hands(n_samples, s_flop, s_turn, s_river, seed, class_of, top5, straight):
    cls = np.empty((n_samples, 2), dtype=np.int32)
    feat = np.empty((n_samples, 3, 2), dtype=np.float32)
    outcome = np.empty(n_samples, dtype=np.int8)
    for i in prange(n_samples):
        np.random.seed(seed + i)
        deck = np.empty(52, dtype=np.int64)
        for c in range(52):
            deck[c] = c
        for k in range(11):
            j = k + np.random.randint(0, 52 - k)
            tmp = deck[k]
            deck[k] = deck[j]
            deck[j] = tmp
        h0 = deck[0:3].copy()
        h1 = deck[3:6].copy()
        board = deck[6:11].copy()
        for p in range(2):
            h = h0 if p == 0 else h1
            a = min(h[0], h[1], h[2])
            c = max(h[0], h[1], h[2])
            b = h[0] + h[1] + h[2] - a - c
            cls[i, p] = class_of[a, b, c]
            feat[i, 0, p] = equity_vs_random(h, board, 3, s_flop, top5, straight)
            feat[i, 1, p] = equity_vs_random(h, board, 4, s_turn, top5, straight)
            feat[i, 2, p] = equity_vs_random(h, board, 5, s_river, top5, straight)
        outcome[i] = showdown(h0, h1, board)
    return cls, feat, outcome


def build_abstraction(
    n_buckets=50,
    n_samples=2_000_000,
    s_flop=500,
    s_turn=500,
    s_river=1000,
    preflop_samples=1000,
    seed=12345,
    log=print,
):
    t0 = time.time()
    log(f"sampling {n_samples:,} deals ...")
    cls, feat, outcome = sample_hands(
        n_samples, s_flop, s_turn, s_river, seed, CLASS_OF, TOP5, STRAIGHT
    )
    log(f"  done in {time.time() - t0:.0f}s")
    K = n_buckets
    edges = np.empty((3, K - 1), dtype=np.float32)
    bucket = np.empty((n_samples, 3, 2), dtype=np.int32)
    for s in range(3):
        pooled = feat[:, s, :].ravel()
        qs = np.quantile(pooled, np.arange(1, K) / K)
        edges[s] = qs
        bucket[:, s, :] = np.searchsorted(qs, feat[:, s, :], side="right")
    # transitions
    def rows(counts):
        counts = counts.astype(np.float64) + 0.01
        return counts / counts.sum(axis=1, keepdims=True)

    t_pre = np.zeros((N_CLASSES, K), dtype=np.int64)
    np.add.at(t_pre, (cls.ravel(), bucket[:, 0, :].ravel()), 1)
    t_flop = np.zeros((K, K), dtype=np.int64)
    np.add.at(t_flop, (bucket[:, 0, :].ravel(), bucket[:, 1, :].ravel()), 1)
    t_turn = np.zeros((K, K), dtype=np.int64)
    np.add.at(t_turn, (bucket[:, 1, :].ravel(), bucket[:, 2, :].ravel()), 1)
    # showdown matrices per postflop street: E[win - loss] for bucket b vs b'
    W = np.zeros((3, K, K), dtype=np.float64)
    for s in range(3):
        b0 = bucket[:, s, 0]
        b1 = bucket[:, s, 1]
        total = np.zeros((K, K))
        count = np.zeros((K, K))
        np.add.at(total, (b0, b1), outcome)
        np.add.at(count, (b0, b1), 1)
        np.add.at(total, (b1, b0), -outcome)
        np.add.at(count, (b1, b0), 1)
        fallback = np.sign(np.arange(K)[:, None] - np.arange(K)[None, :]).astype(np.float64)
        W[s] = np.where(count > 0, total / np.maximum(count, 1), fallback)
        W[s] = 0.5 * (W[s] - W[s].T)  # exactly antisymmetric
    log(f"preflop matchups {N_CLASSES}x{N_CLASSES} x {preflop_samples} ...")
    t1 = time.time()
    win, tie = preflop_matchups(REPRESENTATIVE, preflop_samples, seed + 7)
    w0 = 2.0 * win.astype(np.float64) + tie.astype(np.float64) - 1.0
    w0 = 0.5 * (w0 - w0.T)
    log(f"  done in {time.time() - t1:.0f}s")
    p_class = WEIGHT / WEIGHT.sum()
    return {
        "K": np.int64(K),
        "edges": edges,
        "T_pre": rows(t_pre),
        "T_flop": rows(t_flop),
        "T_turn": rows(t_turn),
        "W0": w0,
        "W_post": W,  # [flop, turn, river]
        "win0": win,
        "tie0": tie,
        "p_class": p_class,
        "meta": np.array([n_samples, s_flop, s_turn, s_river, preflop_samples, seed]),
    }


def bucket_of(edges_street, equity):
    return int(np.searchsorted(edges_street, equity, side="right"))
