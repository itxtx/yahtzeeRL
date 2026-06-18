"""Generate README plots for yahtzeeRL from EVAL_RESULTS.md numbers.

Plot 1: per-category mean score, mcts@step_11800 vs heuristic (512 games).
Plot 4: MCTS win rate vs the greedy policy head across num_simulations.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 12,
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "figure.dpi": 150,
    "savefig.dpi": 150,
})

AGENT = "#2563EB"   # blue  -> learned agent
HEUR  = "#9CA3AF"   # gray  -> heuristic baseline
ACCENT = "#DC2626"  # red   -> reference lines / annotations

# Write the figures next to this script (the repo's plots/ directory).
OUT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Plot 1: per-category means (headline 512-game eval, step_11800 vs heuristic)
# ---------------------------------------------------------------------------
categories = [
    "Ones", "Twos", "Threes", "Fours", "Fives", "Sixes",
    "3 of a Kind", "4 of a Kind", "Full House",
    "Small Straight", "Large Straight", "Yahtzee", "Chance",
]
agent = [1.25, 3.77, 6.86, 9.76, 13.04, 15.89,
         20.16, 15.98, 22.02, 27.95, 33.20, 9.86, 22.07]
heur  = [1.12, 3.38, 5.61, 8.35, 10.20, 13.12,
         18.75, 17.31, 23.10, 8.32, 12.66, 23.24, 18.84]

y = np.arange(len(categories))[::-1]  # first category at top
h = 0.38

fig, ax = plt.subplots(figsize=(9.5, 7.0))
ax.barh(y + h/2, agent, height=h, color=AGENT, label="RL agent (MCTS @ step 11.8k)")
ax.barh(y - h/2, heur,  height=h, color=HEUR,  label="Heuristic baseline")

for yi, a, b in zip(y, agent, heur):
    ax.text(a + 0.4, yi + h/2, f"{a:.1f}", va="center", ha="left",
            fontsize=8.5, color=AGENT)
    ax.text(b + 0.4, yi - h/2, f"{b:.1f}", va="center", ha="left",
            fontsize=8.5, color="#6B7280")

ax.set_yticks(y)
ax.set_yticklabels(categories)
ax.set_xlabel("Mean points per game (512 games, alternating seats)")
ax.set_xlim(0, 38)
ax.set_title("What strategy did self-play learn?\n"
             "Per-category scoring vs a hand-written heuristic",
             fontsize=14, fontweight="bold", loc="left")
ax.text(0, 1.005, "", transform=ax.transAxes)
# subtitle with the headline totals
fig.text(0.012, 0.012,
         "Totals: agent 204.3  vs  heuristic 164.5   |   win rate 81.1%   "
         "(margin +39.8).  Agent wins the straights and the upper section; "
         "it deliberately punts Yahtzee.",
         fontsize=9, color="#374151")
ax.legend(loc="upper right", frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="x", color="#E5E7EB", linewidth=0.8)
ax.set_axisbelow(True)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(f"{OUT}/per_category_agent_vs_heuristic.png", facecolor="white",
            bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Plot 4: MCTS win rate vs greedy policy head, across num_simulations
# Each point = 512 games. 95% binomial CI on the win rate.
# ---------------------------------------------------------------------------
sims = np.array([64, 128, 256])
winrate = np.array([0.494, 0.525, 0.508])  # A=mcts win rate vs greedy
margin = np.array([0.96, 0.26, -0.58])     # mean score margin
n = 512
ci = 1.96 * np.sqrt(winrate * (1 - winrate) / n)

fig, ax = plt.subplots(figsize=(8.5, 5.2))
ax.axhline(0.5, color=ACCENT, linestyle="--", linewidth=1.2, zorder=1)
ax.text(256, 0.502, "parity (0.5)", color=ACCENT, fontsize=9,
        va="bottom", ha="right")

ax.errorbar(sims, winrate, yerr=ci, fmt="o-", color=AGENT, ecolor=AGENT,
            elinewidth=1.4, capsize=5, markersize=9, linewidth=2,
            label="MCTS win rate vs greedy (95% CI, n=512)")

for s, w, m in zip(sims, winrate, margin):
    ax.annotate(f"{w:.3f}\n(margin {m:+.1f})", (s, w),
                textcoords="offset points", xytext=(0, 16),
                ha="center", fontsize=9, color="#374151")

ax.set_xscale("log", base=2)
ax.set_xticks(sims)
ax.set_xticklabels([str(s) for s in sims])
ax.set_xlim(50, 330)
ax.set_ylim(0.44, 0.58)
ax.set_xlabel("MCTS simulations per move")
ax.set_ylabel("Win rate vs the greedy policy head")
ax.set_title("More search buys almost nothing\n"
             "The policy head has already absorbed the search target",
             fontsize=14, fontweight="bold", loc="left")
ax.legend(loc="lower left", frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(f"{OUT}/search_vs_sims.png", facecolor="white", bbox_inches="tight")
plt.close(fig)

print("wrote per_category_agent_vs_heuristic.png and search_vs_sims.png")
