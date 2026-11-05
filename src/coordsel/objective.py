"""Communication-efficiency objective for a set of team coordinators.

    comm_eff(S) = mean_{c in S} closeness_in_team(c)  +  alpha * mean_{c != c'} hops(c,c') / dist(c,c')

* ``closeness_in_team`` is weighted closeness centrality computed inside the
  coordinator's own team, so a coordinator should be central to the people they lead.
* ``hops / dist`` (unweighted path length over weighted shortest-path distance) rewards
  coordinators connected through short, strong ties; it is averaged over every pair of
  coordinators that is connected (pairs with no path are skipped).

All pairwise quantities are pre-computed once, so evaluating a candidate set is O(|S|^2)
lookups. That makes exact search over every one-per-team combination tractable.
"""
from __future__ import annotations

from itertools import combinations

import networkx as nx
import numpy as np

from .networks import TEAMS, team_nodes


class CommEfficiency:
    def __init__(self, G: nx.Graph, alpha: float = 2.0, teams=TEAMS):
        self.G = G
        self.alpha = alpha
        self.teams = list(teams)
        self.nodes = list(G.nodes)
        self.idx = {v: i for i, v in enumerate(self.nodes)}
        self.label = {v: G.nodes[v]["label"] for v in self.nodes}

        n = len(self.nodes)
        self.closeness = np.zeros(n)
        for t in self.teams:
            members = team_nodes(G, t)
            for v, c in nx.closeness_centrality(G.subgraph(members), distance="weight").items():
                self.closeness[self.idx[v]] = c

        hops = dict(nx.all_pairs_shortest_path_length(G))
        dist = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))
        self.ratio = np.full((n, n), np.nan)  # NaN = no path
        for u in self.nodes:
            for v, d in dist[u].items():
                if u != v and d > 0:
                    self.ratio[self.idx[u], self.idx[v]] = hops[u][v] / d

    def in_team(self, S) -> float:
        return float(np.mean([self.closeness[self.idx[v]] for v in S]))

    def cross_team(self, S) -> float:
        vals = [self.ratio[self.idx[u], self.idx[v]] for u, v in combinations(S, 2)]
        vals = [x for x in vals if not np.isnan(x)]
        return float(np.mean(vals)) if vals else 0.0

    def __call__(self, S) -> float:
        S = list(S)
        if not S:
            return 0.0
        return self.alpha * self.cross_team(S) + self.in_team(S)

    def team_members(self, team: str) -> list:
        return [v for v in self.nodes if self.label[v] == team]
