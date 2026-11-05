"""Coordinator-selection algorithms: choose exactly one coordinator per team.

* ``greedy``            seeded greedy: start from a seed, repeatedly add the coordinator
                        (from a team not yet covered) with the best objective value.
* ``best_greedy``       run ``greedy`` from every candidate seed, keep the best.
* ``influence_greedy``  ``best_greedy`` seeded only from each team's most central member
                        (4 greedy runs instead of n).
* ``influence_only``    pick each team's most central member, no interaction modelled.
* ``random_baseline``   Monte Carlo mean (and std) of uniformly random coordinators.
* ``exhaustive``        exact optimum over all prod(|team|) combinations (vectorised).
"""
from __future__ import annotations

import itertools
import random

import numpy as np

from .objective import CommEfficiency


def greedy(f: CommEfficiency, seed) -> tuple[list, float]:
    S = [seed]
    covered = {f.label[seed]}
    while len(covered) < len(f.teams):
        best, best_val = None, -np.inf
        for v in f.nodes:
            if f.label[v] in covered:
                continue
            val = f(S + [v])
            if val > best_val:
                best, best_val = v, val
        S.append(best)
        covered.add(f.label[best])
    return S, f(S)


def best_greedy(f: CommEfficiency, seeds=None) -> tuple[list, float]:
    seeds = f.nodes if seeds is None else seeds
    return max((greedy(f, s) for s in seeds), key=lambda r: r[1])


def most_central(f: CommEfficiency) -> list:
    return [max(f.team_members(t), key=lambda v: f.closeness[f.idx[v]]) for t in f.teams]


def influence_greedy(f: CommEfficiency) -> tuple[list, float]:
    return best_greedy(f, seeds=most_central(f))


def influence_only(f: CommEfficiency) -> tuple[list, float]:
    S = most_central(f)
    return S, f(S)


def random_baseline(f: CommEfficiency, n_iter: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    members = [f.team_members(t) for t in f.teams]
    vals = [f([rng.choice(m) for m in members]) for _ in range(n_iter)]
    return float(np.mean(vals)), float(np.std(vals))


def exhaustive(f: CommEfficiency) -> tuple[list, float, np.ndarray]:
    """Exact optimum. Returns (best set, best value, values of *all* combinations)."""
    members = [np.array([f.idx[v] for v in f.team_members(t)]) for t in f.teams]
    grids = np.meshgrid(*members, indexing="ij")
    combos = np.stack([g.ravel() for g in grids], axis=1)        # (N, n_teams)

    in_team = f.closeness[combos].mean(axis=1)
    pair_vals = np.stack(
        [f.ratio[combos[:, i], combos[:, j]] for i, j in itertools.combinations(range(combos.shape[1]), 2)],
        axis=1)
    with np.errstate(invalid="ignore"):
        cross = np.nanmean(pair_vals, axis=1)
    cross = np.nan_to_num(cross, nan=0.0)
    values = f.alpha * cross + in_team

    best = int(np.argmax(values))
    S = [f.nodes[i] for i in combos[best]]
    return S, float(values[best]), values
