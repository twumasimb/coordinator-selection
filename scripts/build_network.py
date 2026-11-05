"""Build the labelled DBLP expert network and the 20/40/60/80/100-node team networks.

    python scripts/build_network.py --dblp data/raw/dblp.txt --out data/networks
"""
import argparse
import os
import pickle
import sys
from collections import Counter

import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from coordsel.data import (build_author_records, build_expert_network,  # noqa: E402
                           filter_papers, parse_dblp)
from coordsel.networks import build_team_network  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dblp", default="data/raw/dblp.txt")
    ap.add_argument("--out", default="data/networks")
    ap.add_argument("--start-year", type=int, default=2000)
    ap.add_argument("--end-year", type=int, default=2010)
    ap.add_argument("--team-sizes", type=int, nargs="+", default=[5, 10, 15, 20, 25])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    papers = filter_papers(parse_dblp(args.dblp), start_year=args.start_year, end_year=args.end_year)
    print(f"papers in selected venues {args.start_year}-{args.end_year}: {len(papers):,}")
    records = build_author_records(papers)
    print(f"authors with >= 5 papers: {len(records):,}")
    G = build_expert_network(records)
    print(f"expert network: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges, "
          f"{dict(Counter(nx.get_node_attributes(G, 'label').values()))}")
    with open(os.path.join(args.out, "expert_network.pkl"), "wb") as fh:
        pickle.dump(G, fh)

    for k in args.team_sizes:
        H = build_team_network(G, k)
        nx.write_graphml(H, os.path.join(args.out, f"teams_{4 * k}.graphml"))
        print(f"team network k={k}: {H.number_of_nodes()} nodes, {H.number_of_edges()} edges")


if __name__ == "__main__":
    main()
