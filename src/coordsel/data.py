"""Build the labelled DBLP expert network from the AMiner DBLP citation dump.

Pipeline
--------
1. ``parse_dblp``           stream the raw ``#*`` / ``#@`` / ``#t`` / ``#c`` records.
2. ``filter_papers``        keep papers from 19 top venues in a year window.
3. ``build_author_records`` aggregate per-author co-authors, venues and paper counts
                            (authors with >= ``min_papers`` papers only).
4. ``build_expert_network`` weighted author graph (co-authorship + co-venue edges),
                            edge weights min-max scaled to [1, 100].
5. ``assign_label``         label each author with the research area (team) of the
                            venue they publish in most.

Edge weights are *distances*: lower weight = stronger tie.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from typing import Iterator

import networkx as nx

AREA_VENUES: dict[str, list[str]] = {
    "DB": ["VLDB", "ICDE", "ICDT", "EDBT", "PODS", "SIGMOD Conference"],
    "AI": ["ICML", "ECML", "COLT", "UAI"],
    "T": ["SODA", "STOC", "FOCS", "STACS"],
    "DM": ["KDD", "ICDM", "PKDD", "WWW", "SDM"],
}
VENUE_TO_AREA = {v: area for area, venues in AREA_VENUES.items() for v in venues}
ALL_VENUES = list(VENUE_TO_AREA)


def parse_dblp(path: str) -> Iterator[dict]:
    """Yield one dict per paper from the AMiner DBLP-Citation-network V1 text file."""
    paper: dict = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if paper:
                    yield paper
                paper = {}
            elif line.startswith("#*"):
                paper["title"] = line[2:].strip()
            elif line.startswith("#@"):
                paper["authors"] = [a.strip() for a in line[2:].split(",") if a.strip()]
            elif line.startswith("#t"):
                paper["year"] = int(line[2:].strip() or 0)
            elif line.startswith("#c"):
                paper["venue"] = line[2:].strip()
    if paper:
        yield paper


def filter_papers(papers, venues=ALL_VENUES, start_year=2000, end_year=2010) -> list[dict]:
    venues = set(venues)
    return [
        p for p in papers
        if p.get("title") and p.get("authors") and p.get("venue") in venues
        and start_year <= p.get("year", 0) <= end_year
    ]


def build_author_records(papers, min_papers: int = 5) -> dict[str, dict]:
    """Aggregate papers into per-author records.

    ``venue_year`` keeps the *last seen* year per venue (this mirrors the original
    experiment, where two authors are linked if they published at the same venue in
    that same year).
    """
    recs: dict[str, dict] = defaultdict(lambda: {
        "coauthors": Counter(), "titles": set(), "venue_counts": Counter(), "venue_year": {}})
    for p in papers:
        authors = set(p["authors"])
        for a in authors:
            r = recs[a]
            r["coauthors"].update(authors - {a})
            r["titles"].add(p["title"])
            r["venue_counts"][p["venue"]] += 1
            r["venue_year"][p["venue"]] = p["year"]
    out = {}
    for a, r in recs.items():
        if len(r["titles"]) >= min_papers:
            r["num_papers"] = len(r["titles"])
            out[a] = r
    return out


def assign_label(record: dict) -> str | None:
    """Team label = area of the venue where the author published most often."""
    top_venue = max(record["venue_counts"], key=record["venue_counts"].get)
    return VENUE_TO_AREA.get(top_venue)


def scale_edge_weights(G: nx.Graph, lo: float = 1.0, hi: float = 100.0) -> nx.Graph:
    H = G.copy()
    w = [d["weight"] for *_, d in H.edges(data=True)]
    wmin, wmax = min(w), max(w)
    for u, v, d in H.edges(data=True):
        d["weight"] = hi if wmax == wmin else (d["weight"] - wmin) / (wmax - wmin) * (hi - lo) + lo
    return H


def build_expert_network(records: dict[str, dict], min_shared_papers: int = 2) -> nx.Graph:
    """Weighted, labelled expert network.

    * Co-authorship edge (>= ``min_shared_papers`` joint papers):
      ``w = 1 - shared / (papers_u + papers_v)``  (Jaccard-style distance).
    * Co-venue edge: for every venue where both authors' latest paper is in the same
      year, add ``10 * (1 - 2 / (papers_u + papers_v))`` (counted once from each side,
      as in the original experiment).
    Weights are then min-max scaled to [1, 100]; only the largest connected component
    of labelled authors is kept.
    """
    G = nx.Graph()
    n = {a: r["num_papers"] for a, r in records.items()}

    # 1. co-authorship edges. Co-authors below the paper threshold count as 0 papers; they
    #    take part in weight scaling and are dropped afterwards (they carry no label).
    for a, r in records.items():
        for b, shared in r["coauthors"].items():
            if shared >= min_shared_papers and not G.has_edge(a, b):
                G.add_edge(a, b, weight=round(1 - shared / (n[a] + n.get(b, 0)), 4))

    # 2. co-venue edges, via an inverted index (venue, year) -> authors
    index: dict[tuple, list[str]] = defaultdict(list)
    for a, r in records.items():
        for venue_year in r["venue_year"].items():
            index[venue_year].append(a)
    for authors in index.values():
        for a, b in combinations(authors, 2):
            w = 2 * round(1 - 2 / (n[a] + n[b]), 4) * 10
            if G.has_edge(a, b):
                G[a][b]["weight"] += w
            else:
                G.add_edge(a, b, weight=w)

    G.remove_edges_from(nx.selfloop_edges(G))
    G = scale_edge_weights(G, 1, 100)

    for a in G.nodes:
        label = assign_label(records[a]) if a in records else None
        if label:
            G.nodes[a]["label"] = label
    G.remove_nodes_from([a for a in list(G.nodes) if "label" not in G.nodes[a]])
    largest = max(nx.connected_components(G), key=len)
    return G.subgraph(largest).copy()
