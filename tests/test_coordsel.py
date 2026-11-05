import os
import sys

import networkx as nx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from coordsel import (CommEfficiency, apply_project_network, best_greedy,  # noqa: E402
                      exhaustive, influence_only, random_baseline, select_k_members)
from coordsel.data import build_author_records, build_expert_network  # noqa: E402


@pytest.fixture
def toy():
    """Two densely connected nodes per team plus a few cross-team links."""
    G = nx.Graph()
    teams = {"DB": ["d1", "d2"], "T": ["t1", "t2"], "AI": ["a1", "a2"], "DM": ["m1", "m2"]}
    for label, (u, v) in teams.items():
        G.add_node(u, label=label)
        G.add_node(v, label=label)
        G.add_edge(u, v, weight=1.0)
    for u, v, w in [("d1", "t1", 2), ("t1", "a1", 2), ("a1", "m1", 2), ("d2", "m2", 9), ("t2", "a2", 9)]:
        G.add_edge(u, v, weight=w)
    return G


def test_greedy_matches_exhaustive(toy):
    f = CommEfficiency(toy)
    S_g, v_g = best_greedy(f)
    S_o, v_o, values = exhaustive(f)
    assert len(values) == 2 ** 4
    assert v_g == pytest.approx(v_o)          # ties possible, so compare values
    assert f(S_o) == pytest.approx(v_o)
    assert {f.label[v] for v in S_g} == {"DB", "T", "AI", "DM"}


def test_exhaustive_values_match_objective(toy):
    f = CommEfficiency(toy)
    _, _, values = exhaustive(f)
    assert max(values) == pytest.approx(f(best_greedy(f)[0]))


def test_baselines_bounded_by_optimum(toy):
    f = CommEfficiency(toy)
    _, v_opt, _ = exhaustive(f)
    assert influence_only(f)[1] <= v_opt + 1e-12
    assert random_baseline(f, n_iter=200)[0] <= v_opt + 1e-12


def test_project_network_removes_only_unlinked_cross_team_edges(toy):
    H = apply_project_network(toy, "star")  # T is the hub
    assert H.has_edge("d1", "t1") and H.has_edge("t1", "a1")
    assert not H.has_edge("a1", "m1")          # AI-DM not in star
    assert H.has_edge("d1", "d2")              # intra-team edges untouched
    assert apply_project_network(toy, "full").number_of_edges() == toy.number_of_edges()


def test_select_k_members_is_nested():
    G = nx.path_graph(10)
    nx.set_edge_attributes(G, 1.0, "weight")
    assert select_k_members(G, 6)[:3] == select_k_members(G, 3)


def test_build_expert_network_labels_and_weights():
    papers = []
    for i in range(6):
        papers.append({"title": f"db{i}", "authors": ["Ann", "Bob"], "venue": "VLDB", "year": 2005})
        papers.append({"title": f"ai{i}", "authors": ["Cat", "Bob"], "venue": "ICML", "year": 2006})
    papers += [{"title": f"x{i}", "authors": ["Cat"], "venue": "UAI", "year": 2007} for i in range(4)]
    recs = build_author_records(papers, min_papers=5)
    G = build_expert_network(recs)
    assert G.nodes["Ann"]["label"] == "DB"
    assert G.nodes["Cat"]["label"] == "AI"
    w = [d["weight"] for *_, d in G.edges(data=True)]
    assert min(w) == pytest.approx(1.0) and max(w) == pytest.approx(100.0)
