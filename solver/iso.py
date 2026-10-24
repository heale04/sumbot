"""Suit-isomorphism classes for three-card preflop hands.

Two hands are in the same class if one maps to the other by permuting suits. CLASS_OF maps a
sorted card triple (a < b < c) to its class id; REPRESENTATIVE gives one combo per class and
WEIGHT counts the raw combos in each class (sums to 22100).
"""

import itertools

import numpy as np

PERMS = list(itertools.permutations(range(4)))


def canonical(cards):
    best = None
    for perm in PERMS:
        mapped = tuple(sorted((c >> 2) * 4 + perm[c & 3] for c in cards))
        if best is None or mapped < best:
            best = mapped
    return best


def _build():
    class_of = np.full((52, 52, 52), -1, dtype=np.int16)
    keys = {}
    reps = []
    weights = []
    for combo in itertools.combinations(range(52), 3):
        key = canonical(combo)
        if key not in keys:
            keys[key] = len(reps)
            reps.append(key)
            weights.append(0)
        cid = keys[key]
        weights[cid] += 1
        class_of[combo[0], combo[1], combo[2]] = cid
    return class_of, np.array(reps, dtype=np.int64), np.array(weights, dtype=np.int64)


CLASS_OF, REPRESENTATIVE, WEIGHT = _build()
N_CLASSES = len(REPRESENTATIVE)


def class_id(cards) -> int:
    a, b, c = sorted(int(x) for x in cards)
    return int(CLASS_OF[a, b, c])


def class_name(cid: int) -> str:
    from solver.cards import card_to_str

    return "".join(card_to_str(c) for c in sorted(REPRESENTATIVE[cid], reverse=True))
