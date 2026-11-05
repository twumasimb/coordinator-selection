"""Experiment networks: k-member teams and project-network topologies."""
from __future__ import annotations

from itertools import combinations

import networkx as nx

TEAMS = ["DB", "T", "AI", "DM"]

# Project networks: which teams must communicate with each other.
PROJECT_NETWORKS: dict[str, list[tuple[str, str]]] = {
    # Theory team is the hub every other team talks to.
    "star": [("DM", "T"), ("DB", "T"), ("AI", "T")],
    # DM - AI - T - DB
    "chain": [("DM", "AI"), ("AI", "T"), ("DB", "T")],
    # every team talks to every other team
    "full": list(combinations(TEAMS, 2)),
}


def team_nodes(G: nx.Graph, label: str) -> list:
    return [v for v, d in G.nodes(data=True) if d.get("label") == label]


def select_k_members(G: nx.Graph, k: int) -> list:
    """Greedily pick a tightly-knit k-member team from one area.

    Seed with the member of highest (weighted) closeness centrality, then repeatedly add
    the node with the smallest sum of shortest-path distances to the members chosen so
    far. The selection is nested: the k=5 team is a prefix of the k=10 team, etc.
    """
    if k > G.number_of_nodes():
        raise ValueError("k is larger than the team's pool of experts")
    closeness = nx.closeness_centrality(G, distance="weight")
    selected = [max(closeness, key=closeness.get)]
    dist_sum = dict.fromkeys(G.nodes, 0.0)
    while len(selected) < k:
        last = selected[-1]
        lengths = nx.single_source_dijkstra_path_length(G, last, weight="weight")
        for v in dist_sum:
            dist_sum[v] += lengths.get(v, float("inf"))
        best = min((v for v in G.nodes if v not in selected), key=lambda v: dist_sum[v])
        selected.append(best)
    return selected


def build_team_network(G: nx.Graph, k: int, teams=TEAMS) -> nx.Graph:
    """Sub-network with ``k`` experts from each team (``4k`` nodes in total)."""
    nodes = []
    for label in teams:
        nodes += select_k_members(G.subgraph(team_nodes(G, label)), k)
    return G.subgraph(nodes).copy()


def apply_project_network(G: nx.Graph, topology: str) -> nx.Graph:
    """Drop cross-team edges between teams that are not linked in the project network."""
    P = nx.Graph(PROJECT_NETWORKS[topology])
    H = G.copy()
    H.remove_edges_from([
        (u, v) for u, v in G.edges
        if G.nodes[u]["label"] != G.nodes[v]["label"]
        and not P.has_edge(G.nodes[u]["label"], G.nodes[v]["label"])
    ])
    return H
