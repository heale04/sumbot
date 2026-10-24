"""Vectorized CFR+ for the N-player preflop abstract game (multiway equity showdown model).

Each player's hand is an independent preflop bucket (clusters of suit-isomorphism classes).
At a showdown terminal, player p with bucket b wins the pot with probability
prod over live opponents q of (Pw @ r_q)[b] / sum(r_q), where Pw[b, b'] = P(b beats b')
with ties counted half; the product treats opponents' hands as independent given b.
"""

import numpy as np
from numba import njit
from sklearn.cluster import KMeans

from solver.multi_tree import DECISION, FOLD, SHOWDOWN


def cluster_preflop(win0, tie0, p_class, n_buckets, seed=0):
    """K-means over matchup profiles. Returns (bucket_of_class, Pw_bucket, p_bucket)."""
    pw = win0.astype(np.float64) + 0.5 * tie0.astype(np.float64)
    feats = pw  # row i: how class i fares against every class
    km = KMeans(n_clusters=n_buckets, n_init=4, random_state=seed)
    labels = km.fit_predict(feats, sample_weight=p_class)
    # order buckets by equity vs the field so bucket index is monotone in strength
    eq = pw @ p_class
    means = np.array([np.average(eq[labels == k], weights=p_class[labels == k]) for k in range(n_buckets)])
    order = np.argsort(means)
    remap = np.empty(n_buckets, dtype=np.int64)
    remap[order] = np.arange(n_buckets)
    labels = remap[labels]
    M = np.zeros((n_buckets, len(p_class)))
    M[labels, np.arange(len(p_class))] = p_class
    p_bucket = M.sum(axis=1)
    Mn = M / p_bucket[:, None]
    Pw_bucket = Mn @ pw @ Mn.T
    return labels.astype(np.int64), Pw_bucket, p_bucket


class MultiSolver:
    def __init__(self, tree, Pw, p_bucket, avg_weight="quad", pos_alpha=0.12, agg_beta=0.10):
        arrs = tree.arrays()
        self.tree = tree
        self.N = tree.config.players
        for k, v in arrs.items():
            setattr(self, k, v)
        # equity realization factor per (terminal, player): in position (last live player) +alpha,
        # out of position -alpha, last preflop aggressor +beta; folded players unused
        n = len(self.node_type)
        self.term_real = np.ones((n, self.N))
        for i in range(n):
            if self.node_type[i] != SHOWDOWN:
                continue
            live = [q for q in range(self.N) if (int(self.term_live[i]) >> q) & 1]
            ip = max(live)
            if all(self.term_contrib[i, q] >= tree.config.stack for q in live):
                continue  # all-in showdown: raw equity, nothing left to realize
            for q in live:
                self.term_real[i, q] = 1.0 + (pos_alpha if q == ip else -pos_alpha) + (agg_beta if q == self.term_aggressor[i] else 0.0)
        self.K = len(p_bucket)
        n = len(self.node_type)
        self.ioff = np.zeros(n + 1, dtype=np.int64)
        for i in range(n):
            self.ioff[i + 1] = self.ioff[i] + (self.K * self.n_actions[i] if self.node_type[i] == DECISION else 0)
        self.I = int(self.ioff[-1])
        self.V = n * self.K
        self.theta = np.zeros(self.I)
        self.strat = np.zeros(self.I)
        self.strat_sum = np.zeros(self.I)
        self.reach = np.zeros((self.N, self.V))
        self.val = np.zeros(self.V)
        self.Pw = np.ascontiguousarray(Pw, dtype=np.float64)
        self.p_bucket = np.ascontiguousarray(p_bucket, dtype=np.float64)
        self.avg_weight = {"const": 0, "lin": 1, "quad": 2}[avg_weight]
        self.t = 0
        m_strategy_from_theta(self.node_type, self.n_actions, self.ioff, self.K, self.theta, self.strat)

    def iterate(self, n_iter=1):
        for _ in range(n_iter):
            self.t += 1
            w = float(self.t) ** self.avg_weight
            for p in range(self.N):
                m_forward(self.node_type, self.player, self.child, self.n_actions, self.ioff, self.K, self.strat,
                          self.reach, self.p_bucket)
                m_backward(p, w, self.node_type, self.player, self.child, self.n_actions, self.ioff, self.K,
                           self.term_pot, self.term_contrib, self.term_live, self.term_real, self.strat, self.reach, self.val,
                           self.theta, self.strat_sum, self.Pw)
                m_strategy_from_theta_player(p, self.node_type, self.player, self.n_actions, self.ioff, self.K,
                                             self.theta, self.strat)

    def average_strategy(self):
        out = np.zeros(self.I)
        m_strategy_from_theta(self.node_type, self.n_actions, self.ioff, self.K, self.strat_sum, out)
        return out

    def nash_conv(self):
        """sum_p (best response value of p vs the others' average strategy - p's value)."""
        sigma = self.average_strategy()
        m_forward(self.node_type, self.player, self.child, self.n_actions, self.ioff, self.K, sigma,
                  self.reach, self.p_bucket)
        total = 0.0
        per = []
        for p in range(self.N):
            vb = m_best_response(p, True, self.node_type, self.player, self.child, self.n_actions, self.ioff, self.K,
                                 self.term_pot, self.term_contrib, self.term_live, self.term_real, sigma, self.reach, self.val, self.Pw)
            ve = m_best_response(p, False, self.node_type, self.player, self.child, self.n_actions, self.ioff, self.K,
                                 self.term_pot, self.term_contrib, self.term_live, self.term_real, sigma, self.reach, self.val, self.Pw)
            gain = float(np.dot(self.p_bucket, vb) - np.dot(self.p_bucket, ve))
            per.append(gain)
            total += gain
        return total, per


@njit(cache=False)
def m_strategy_from_theta(node_type, n_actions, ioff, K, theta, strat):
    n = len(node_type)
    for i in range(n):
        if node_type[i] != DECISION:
            continue
        A = n_actions[i]
        base = ioff[i]
        for b in range(K):
            s = 0.0
            for a in range(A):
                s += theta[base + b * A + a]
            for a in range(A):
                strat[base + b * A + a] = theta[base + b * A + a] / s if s > 0.0 else 1.0 / A


@njit(cache=False)
def m_strategy_from_theta_player(p, node_type, player, n_actions, ioff, K, theta, strat):
    n = len(node_type)
    for i in range(n):
        if node_type[i] != DECISION or player[i] != p:
            continue
        A = n_actions[i]
        base = ioff[i]
        for b in range(K):
            s = 0.0
            for a in range(A):
                s += theta[base + b * A + a]
            for a in range(A):
                strat[base + b * A + a] = theta[base + b * A + a] / s if s > 0.0 else 1.0 / A


@njit(cache=False)
def m_forward(node_type, player, child, n_actions, ioff, K, strat, reach, p_bucket):
    n = len(node_type)
    N = reach.shape[0]
    for q in range(N):
        for b in range(K):
            reach[q, b] = p_bucket[b]
    for i in range(n):
        if node_type[i] != DECISION:
            continue
        A = n_actions[i]
        p = player[i]
        base = ioff[i]
        vb = i * K
        for a in range(A):
            c = child[i, a]
            vc = c * K
            for q in range(N):
                if q == p:
                    for b in range(K):
                        reach[q, vc + b] = reach[q, vb + b] * strat[base + b * A + a]
                else:
                    for b in range(K):
                        reach[q, vc + b] = reach[q, vb + b]


@njit(cache=False)
def m_terminal(i, p, K, term_pot, term_contrib, term_live, term_real, reach, val, Pw, prod):
    N = reach.shape[0]
    vb = i * K
    M = 1.0
    for q in range(N):
        if q == p:
            continue
        tot = 0.0
        for b in range(K):
            tot += reach[q, vb + b]
        if q < 64 and (term_live[i] >> q) & 1:
            if tot <= 0.0:
                M = 0.0
                break
            M *= tot
        else:
            M *= tot
    cp = term_contrib[i, p]
    if M <= 0.0:
        for b in range(K):
            val[vb + b] = 0.0
        return
    if not ((term_live[i] >> p) & 1):
        for b in range(K):
            val[vb + b] = -cp * M
        return
    for b in range(K):
        prod[b] = 1.0
    for q in range(N):
        if q == p or not ((term_live[i] >> q) & 1):
            continue
        tot = 0.0
        for b in range(K):
            tot += reach[q, vb + b]
        rq = reach[q, vb:vb + K] / tot
        pw = np.dot(Pw, rq)
        for b in range(K):
            prod[b] *= pw[b]
    pot = term_pot[i]
    rp = term_real[i, p]
    rbar = 0.0
    nl = 0
    for q in range(N):
        if q != p and ((term_live[i] >> q) & 1):
            rbar += term_real[i, q]
            nl += 1
    rbar = rbar / nl if nl > 0 else 1.0
    for b in range(K):
        pw = prod[b]
        adj = rp * pw / (rp * pw + rbar * (1.0 - pw))
        val[vb + b] = M * (pot * adj - cp)


@njit(cache=False)
def m_backward(p, w, node_type, player, child, n_actions, ioff, K, term_pot, term_contrib, term_live, term_real, strat,
               reach, val, theta, strat_sum, Pw):
    n = len(node_type)
    prod = np.empty(K)
    u = np.empty(16)
    for i in range(n - 1, -1, -1):
        nt = node_type[i]
        vb = i * K
        if nt != DECISION:
            m_terminal(i, p, K, term_pot, term_contrib, term_live, term_real, reach, val, Pw, prod)
            continue
        A = n_actions[i]
        base = ioff[i]
        if player[i] != p:
            for b in range(K):
                acc = 0.0
                for a in range(A):
                    acc += val[child[i, a] * K + b]
                val[vb + b] = acc
            continue
        for b in range(K):
            idx0 = base + b * A
            node_v = 0.0
            for a in range(A):
                u[a] = val[child[i, a] * K + b]
                node_v += strat[idx0 + a] * u[a]
            wt = w * reach[p, vb + b]
            for a in range(A):
                strat_sum[idx0 + a] += wt * strat[idx0 + a]
                r = theta[idx0 + a] + u[a] - node_v
                theta[idx0 + a] = r if r > 0.0 else 0.0
            val[vb + b] = node_v


@njit(cache=False)
def m_best_response(p, maximize, node_type, player, child, n_actions, ioff, K, term_pot, term_contrib, term_live,
                    term_real, strat, reach, val, Pw):
    n = len(node_type)
    prod = np.empty(K)
    for i in range(n - 1, -1, -1):
        nt = node_type[i]
        vb = i * K
        if nt != DECISION:
            m_terminal(i, p, K, term_pot, term_contrib, term_live, term_real, reach, val, Pw, prod)
            continue
        A = n_actions[i]
        base = ioff[i]
        if player[i] != p:
            for b in range(K):
                acc = 0.0
                for a in range(A):
                    acc += val[child[i, a] * K + b]
                val[vb + b] = acc
        elif maximize:
            for b in range(K):
                best = -1e300
                for a in range(A):
                    v = val[child[i, a] * K + b]
                    if v > best:
                        best = v
                val[vb + b] = best
        else:
            for b in range(K):
                acc = 0.0
                for a in range(A):
                    acc += strat[base + b * A + a] * val[child[i, a] * K + b]
                val[vb + b] = acc
    return val[0:K].copy()
