"""Malmuth-Harville ICM (same algorithm as poker_engine.scoring.icm, copied so the bot has no
dependency on the engine package)."""


def icm(stacks, payouts):
    players = [p for p in range(len(stacks)) if stacks[p] > 0]
    total = sum(stacks[p] for p in players)
    equity = [0.0] * len(stacks)
    places = min(len(payouts), len(players))
    chance = {0: 1.0}
    for place in range(places):
        following = {}
        for mask, probability in chance.items():
            remaining = total - sum(stacks[p] for i, p in enumerate(players) if mask >> i & 1)
            for i, player in enumerate(players):
                if mask >> i & 1:
                    continue
                share = probability * stacks[player] / remaining
                equity[player] += share * payouts[place]
                following[mask | 1 << i] = following.get(mask | 1 << i, 0.0) + share
        chance = following
    return equity
