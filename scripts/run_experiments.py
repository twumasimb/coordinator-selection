"""Run every coordinator-selection method on every (network size x project topology).

    python scripts/run_experiments.py --networks data/networks --out results
"""
import argparse
import glob
import json
import os
import re
import sys
import time

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from coordsel import (PROJECT_NETWORKS, CommEfficiency, apply_project_network,  # noqa: E402
                      best_greedy, exhaustive, influence_greedy, influence_only,
                      random_baseline)


def timed(fn, *a, **kw):
    t = time.perf_counter()
    out = fn(*a, **kw)
    return out, time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--networks", default="data/networks")
    ap.add_argument("--out", default="results")
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--random-iters", type=int, default=1000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.networks, "teams_*.graphml")),
                   key=lambda p: int(re.findall(r"\d+", os.path.basename(p))[0]))
    rows, selections = [], {}
    for path in files:
        base = nx.read_graphml(path)
        size = base.number_of_nodes()
        for topo in PROJECT_NETWORKS:
            G = apply_project_network(base, topo)
            f = CommEfficiency(G, alpha=args.alpha)
            (S_opt, v_opt, all_vals), t_opt = timed(exhaustive, f)
            (rnd_mean, rnd_std), t_rnd = timed(random_baseline, f, args.random_iters)
            methods = {
                "Greedy (all seeds)": timed(best_greedy, f),
                "Influence Greedy": timed(influence_greedy, f),
                "Influence only": timed(influence_only, f),
            }
            base_row = dict(nodes=size, topology=topo, edges=G.number_of_edges(),
                            combinations=len(all_vals), optimum=v_opt, random_mean=rnd_mean)
            rows.append({**base_row, "method": "Exhaustive (optimal)", "value": v_opt,
                         "pct_of_opt": 100.0, "gain_vs_random_pct": 100 * (v_opt / rnd_mean - 1),
                         "percentile": 100.0, "seconds": t_opt})
            rows.append({**base_row, "method": "Random (Monte Carlo)", "value": rnd_mean,
                         "value_std": rnd_std, "pct_of_opt": 100 * rnd_mean / v_opt,
                         "gain_vs_random_pct": 0.0,
                         "percentile": 100 * float(np.mean(all_vals <= rnd_mean)), "seconds": t_rnd})
            for name, ((S, v), secs) in methods.items():
                rows.append({**base_row, "method": name, "value": v,
                             "pct_of_opt": 100 * v / v_opt,
                             "gain_vs_random_pct": 100 * (v / rnd_mean - 1),
                             "percentile": 100 * float(np.mean(all_vals <= v + 1e-12)),
                             "seconds": secs})
                selections[f"{size}/{topo}/{name}"] = sorted(S)
            selections[f"{size}/{topo}/Exhaustive (optimal)"] = sorted(S_opt)
            print(f"{size:>3} nodes | {topo:5} | opt {v_opt:.4f} | greedy "
                  f"{rows[-3]['value']:.4f} ({rows[-3]['pct_of_opt']:.1f}%) | "
                  f"random {rnd_mean:.4f} | {len(all_vals):,} combos in {t_opt:.2f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "results.csv"), index=False)
    with open(os.path.join(args.out, "selected_coordinators.json"), "w") as fh:
        json.dump(selections, fh, indent=2, ensure_ascii=False)
    print(f"\nwrote {len(df)} rows to {args.out}/results.csv")


if __name__ == "__main__":
    main()
