"""Plot results/results.csv -> results/figures/*.png and a markdown table."""
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

# categorical slots 1-4 of the validated default palette (fixed order)
COLORS = {"Greedy (all seeds)": "#2a78d6", "Influence Greedy": "#eb6834",
          "Influence only": "#1baf7a", "Random (Monte Carlo)": "#eda100"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
TOPOS = ["star", "chain", "full"]


def style(ax):
    ax.set_facecolor("#fcfcfb")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/results.csv")
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.results)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True, facecolor="#fcfcfb")
    for ax, topo in zip(axes, TOPOS):
        style(ax)
        d = df[df.topology == topo]
        opt = d[d.method == "Exhaustive (optimal)"].sort_values("nodes")
        ax.plot(opt.nodes, opt.value, color=INK, lw=1, ls="--", zorder=5)
        for m, c in COLORS.items():
            s = d[d.method == m].sort_values("nodes")
            if m == "Greedy (all seeds)":  # wide underlay so overlap with others stays visible
                ax.plot(s.nodes, s.value, color=c, lw=7, alpha=0.35, solid_capstyle="round",
                        label=m, zorder=2)
                continue
            ax.plot(s.nodes, s.value, color=c, lw=2, marker="o", ms=6,
                    markeredgecolor="#fcfcfb", markeredgewidth=1.5, label=m, zorder=3)
            if m == "Random (Monte Carlo)":
                ax.fill_between(s.nodes, s.value - s.value_std, s.value + s.value_std,
                                color=c, alpha=0.15, lw=0)
        ax.set_title(f"{topo} project network", color=INK, fontsize=11, loc="left")
        ax.set_xticks(sorted(d.nodes.unique()))
        ax.set_xlabel("expert-network size (nodes)", color=MUTED, fontsize=9)
    axes[0].set_ylabel("communication efficiency", color=MUTED, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=INK, lw=1, ls="--"))
    labels.append("Exact optimum (exhaustive)")
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=9,
               labelcolor=INK, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(args.out, "efficiency_vs_size.png"), dpi=160, bbox_inches="tight")

    # markdown table: greedy vs random vs optimum
    piv = []
    for (n, t), d in df.groupby(["nodes", "topology"]):
        g = d.set_index("method")
        piv.append({"Nodes": n, "Project net": t, "Combinations": f"{int(g.combinations.iloc[0]):,}",
                    "Optimum": f"{g.loc['Exhaustive (optimal)', 'value']:.4f}",
                    "Greedy": f"{g.loc['Greedy (all seeds)', 'value']:.4f}",
                    "Influence Greedy": f"{g.loc['Influence Greedy', 'value']:.4f}",
                    "Influence only": f"{g.loc['Influence only', 'value']:.4f}",
                    "Random (mean)": f"{g.loc['Random (Monte Carlo)', 'value']:.4f}",
                    "Greedy vs random": f"+{g.loc['Greedy (all seeds)', 'gain_vs_random_pct']:.1f}%"})
    tab = pd.DataFrame(piv)
    tab["Project net"] = pd.Categorical(tab["Project net"], TOPOS)
    tab = tab.sort_values(["Project net", "Nodes"])
    with open(os.path.join(os.path.dirname(args.out), "results_table.md"), "w") as fh:
        fh.write(tab.to_markdown(index=False))
    print(tab.to_markdown(index=False))


if __name__ == "__main__":
    main()
