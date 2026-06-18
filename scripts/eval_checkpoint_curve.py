#!/usr/bin/env python
"""Sweep every saved checkpoint against the fixed heuristic and plot the curve.

For each ``step_*`` checkpoint under ``--checkpoint-root`` this runs

    python -m yahtzee_rl.evaluate \
      --agent-a {greedy|mcts} --checkpoint-a <step_dir> \
      --agent-b heuristic --num-games N [--sims-a S] --seed K

as a subprocess (matching the project's "run evaluation through subprocesses"
pattern so JAX's compiled programs and allocator pool are not kept alive across
checkpoints), parses the printed summary into a CSV, and plots win rate / mean
score / margin versus training step. The best win-rate checkpoint is annotated
automatically; if the final checkpoint regressed from that peak, the drop is
annotated too.

Use the greedy head for the curve: per EVAL_RESULTS.md, MCTS buys essentially
nothing over greedy at evaluation time, and greedy is far cheaper (no tree
search), so it gives the same peak/regression shape across ~100 checkpoints in a
fraction of the time. Reserve a 512-game/32-sim MCTS eval for confirming the
single best checkpoint.

Examples
--------
Greedy sweep over a run split across several folders (merged into one curve):

    python scripts/eval_checkpoint_curve.py \
      --checkpoint-root checkpoints/seg1 checkpoints/seg2 checkpoints/seg3 checkpoints/seg4 \
      --agent greedy --num-games 256 --annotate-step 11800 12800

Confirm just the best checkpoint with the headline 512-game/32-sim MCTS eval:

    python -m yahtzee_rl.evaluate --agent-a mcts \
      --checkpoint-a checkpoints/seg4/step_011800 --agent-b heuristic \
      --num-games 512 --sims-a 32

Re-draw the figure from an existing CSV without re-running any evaluations:

    python scripts/eval_checkpoint_curve.py --plot-only --csv plots/winrate_vs_step.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Palette shared with plots/make_plots.py for a consistent README look.
AGENT = "#2563EB"   # blue  -> learned agent
HEUR = "#9CA3AF"    # gray  -> heuristic baseline
ACCENT = "#DC2626"  # red   -> reference lines / annotations

CSV_FIELDS = [
    "step", "checkpoint", "agent", "sims", "num_games",
    "a_win_rate", "b_win_rate", "draw_rate",
    "mean_a_score", "mean_b_score", "mean_margin", "ci95",
]

_STEP_DIR = re.compile(r"^step_(\d+)$")
_RE_STEP = re.compile(r"Agent A:\s*\S+@step_(\d+)")
_RE_GAMES = re.compile(r"games:\s*(\d+)")
_RE_WIN = re.compile(
    r"A win:\s*([\d.]+)\s*\|\s*B win:\s*([\d.]+)\s*\|\s*draw:\s*([\d.]+)"
)
_RE_SCORE = re.compile(
    r"mean score A:\s*(-?[\d.]+)\s*\|\s*B:\s*(-?[\d.]+)\s*\|\s*margin:\s*(-?[\d.]+)"
)


# ---------------------------------------------------------------------------
# Discovery + parsing (kept import-free so they are unit-testable)
# ---------------------------------------------------------------------------
def find_checkpoints(root: str) -> list[tuple[int, str]]:
    """Return [(step, path), ...] for ``step_<digits>`` dirs under ``root``."""
    if not os.path.isdir(root):
        raise SystemExit(f"checkpoint root not found: {root}")
    found = []
    for name in os.listdir(root):
        match = _STEP_DIR.match(name)
        path = os.path.join(root, name)
        if match and os.path.isdir(path):
            found.append((int(match.group(1)), path))
    return sorted(found, key=lambda item: item[0])


def gather_checkpoints(roots: list[str]) -> list[tuple[int, str]]:
    """Merge checkpoints from several roots into one step-sorted list.

    Useful when a long run is split across folders (e.g. resume segments).
    Steps are deduped: if the same step appears in two roots the later root on
    the command line wins, and a warning is printed.
    """
    by_step: dict[int, str] = {}
    for root in roots:
        for step, path in find_checkpoints(root):
            if step in by_step and by_step[step] != path:
                print(f"  ! duplicate step {step}: using {path} "
                      f"(was {by_step[step]})")
            by_step[step] = path
    return sorted(by_step.items(), key=lambda item: item[0])


def parse_summary(text: str) -> dict | None:
    """Parse the stdout of ``yahtzee_rl.evaluate`` into a metrics dict.

    Returns ``None`` if the required win/score lines are missing.
    """
    win = _RE_WIN.search(text)
    score = _RE_SCORE.search(text)
    if not win or not score:
        return None
    games = _RE_GAMES.search(text)
    step = _RE_STEP.search(text)
    return {
        "step": int(step.group(1)) if step else None,
        "num_games": int(games.group(1)) if games else None,
        "a_win_rate": float(win.group(1)),
        "b_win_rate": float(win.group(2)),
        "draw_rate": float(win.group(3)),
        "mean_a_score": float(score.group(1)),
        "mean_b_score": float(score.group(2)),
        "mean_margin": float(score.group(3)),
    }


def ci95(p: float, n: int | None) -> float:
    """95% normal-approx binomial half-width for a win rate ``p`` over ``n`` games."""
    if not n:
        return 0.0
    return 1.96 * (p * (1.0 - p) / n) ** 0.5


# ---------------------------------------------------------------------------
# Running the sweep
# ---------------------------------------------------------------------------
def build_command(agent: str, checkpoint: str, opponent: str,
                  num_games: int, sims: int, seed: int) -> list[str]:
    cmd = [
        sys.executable, "-m", "yahtzee_rl.evaluate",
        "--agent-a", agent, "--checkpoint-a", checkpoint,
        "--agent-b", opponent,
        "--num-games", str(num_games),
        "--seed", str(seed),
    ]
    if agent == "mcts":
        cmd += ["--sims-a", str(sims)]
    return cmd


def subsample(checkpoints: list[tuple[int, str]], every: int,
              keep_steps: list[int]) -> list[tuple[int, str]]:
    """Keep every Nth checkpoint, always retaining the first, last, and any
    explicitly requested steps (so a coarse sweep never drops the peak)."""
    if every <= 1:
        return checkpoints
    keep = set(keep_steps)
    last = len(checkpoints) - 1
    return [cp for i, cp in enumerate(checkpoints)
            if i % every == 0 or i == last or cp[0] in keep]


def run_sweep(args) -> list[dict]:
    checkpoints = gather_checkpoints(args.checkpoint_root)
    if not checkpoints:
        roots = ", ".join(args.checkpoint_root)
        raise SystemExit(f"no step_* checkpoints under: {roots}")
    total = len(checkpoints)
    checkpoints = subsample(checkpoints, args.every, args.annotate_step)
    note = f" (every {args.every} -> {len(checkpoints)} kept)" if args.every > 1 else ""
    print(f"Found {total} checkpoints across "
          f"{len(args.checkpoint_root)} root(s){note}")

    rows: list[dict] = []
    for step, path in checkpoints:
        cmd = build_command(args.agent, path, args.opponent,
                            args.num_games, args.sims, args.seed)
        if args.dry_run:
            print("DRY RUN:", " ".join(cmd))
            continue
        print(f"[step {step}] {args.agent} vs {args.opponent} "
              f"({args.num_games} games)...", flush=True)
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"  ! evaluate failed (exit {proc.returncode}); skipping\n"
                  f"  {proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else ''}")
            continue
        summary = parse_summary(proc.stdout)
        if summary is None:
            print("  ! could not parse evaluate output; skipping")
            continue
        row = {
            "step": summary["step"] if summary["step"] is not None else step,
            "checkpoint": path,
            "agent": args.agent,
            "sims": args.sims if args.agent == "mcts" else "",
            "num_games": summary["num_games"],
            "a_win_rate": summary["a_win_rate"],
            "b_win_rate": summary["b_win_rate"],
            "draw_rate": summary["draw_rate"],
            "mean_a_score": summary["mean_a_score"],
            "mean_b_score": summary["mean_b_score"],
            "mean_margin": summary["mean_margin"],
            "ci95": ci95(summary["a_win_rate"], summary["num_games"]),
        }
        rows.append(row)
        print(f"  win {row['a_win_rate']:.3f}  "
              f"score {row['mean_a_score']:.1f} vs {row['mean_b_score']:.1f}  "
              f"margin {row['mean_margin']:+.1f}")
        write_csv(args.csv, rows)  # write incrementally so a long sweep is durable
    return rows


def write_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: str) -> list[dict]:
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        for key in ("step", "num_games"):
            row[key] = int(row[key]) if row[key] not in ("", None) else None
        for key in ("a_win_rate", "b_win_rate", "draw_rate",
                    "mean_a_score", "mean_b_score", "mean_margin", "ci95"):
            row[key] = float(row[key]) if row[key] not in ("", None) else None
    return rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_curve(rows: list[dict], out_path: str, opponent: str,
               annotate_steps: list[int]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 12,
        "font.family": "DejaVu Sans",
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "figure.dpi": 150,
        "savefig.dpi": 150,
    })

    rows = sorted(rows, key=lambda r: r["step"])
    steps = [r["step"] for r in rows]
    win = [r["a_win_rate"] for r in rows]
    ci = [r["ci95"] for r in rows]
    a_score = [r["mean_a_score"] for r in rows]
    b_score = [r["mean_b_score"] for r in rows]
    agent_label = rows[0]["agent"] if rows else "agent"
    games = next((r["num_games"] for r in rows if r["num_games"]), None)
    n_note = f", n={games}/checkpoint" if games else ""

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(9.5, 8.4), sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": [1.0, 1.0]},
    )

    # --- Panel 1: win rate vs step, with 95% CI band and parity line --------
    lo = [w - c for w, c in zip(win, ci)]
    hi = [w + c for w, c in zip(win, ci)]
    # Headroom above the peak so the annotations clear the curve and the title.
    ax_top.set_ylim(min(min(lo), 0.48) - 0.02, max(hi) + 0.12)
    ax_top.fill_between(steps, lo, hi, color=AGENT, alpha=0.15, linewidth=0,
                        label=f"95% CI{n_note}")
    ax_top.plot(steps, win, "o-", color=AGENT, linewidth=2, markersize=6,
                label=f"{agent_label} win rate vs {opponent}")
    ax_top.axhline(0.5, color=ACCENT, linestyle="--", linewidth=1.2, zorder=1)
    ax_top.text(steps[-1], 0.505, "parity (0.5)", color=ACCENT, fontsize=9,
                va="bottom", ha="right")
    ax_top.set_ylabel(f"Win rate vs {opponent}")
    ax_top.set_title(
        "Competitive strength across training\n"
        f"{agent_label} vs the fixed {opponent} baseline, per saved checkpoint",
        fontsize=14, fontweight="bold", loc="left", pad=12)
    ax_top.legend(loc="lower right", frameon=False)
    ax_top.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax_top.set_axisbelow(True)
    for spine in ("top", "right"):
        ax_top.spines[spine].set_visible(False)

    # Auto-annotate the peak win-rate checkpoint. Place the label toward the
    # open side of the axis so it clears the title and the right margin.
    peak_i = max(range(len(win)), key=lambda i: win[i])
    frac = peak_i / max(len(steps) - 1, 1)
    if frac > 0.6:
        peak_dx, peak_ha = -60, "right"
    elif frac < 0.4:
        peak_dx, peak_ha = 60, "left"
    else:
        peak_dx, peak_ha = 0, "center"
    ax_top.annotate(
        f"peak: step {steps[peak_i]}\nwin {win[peak_i]:.3f}",
        (steps[peak_i], win[peak_i]),
        textcoords="offset points", xytext=(peak_dx, 22), ha=peak_ha,
        fontsize=9, color="#111827", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#111827", lw=1.0))
    # If the last checkpoint dropped meaningfully below the peak, flag it.
    if len(win) > 1 and peak_i != len(win) - 1 and win[-1] < win[peak_i] - 0.01:
        ax_top.annotate(
            f"regression\nstep {steps[-1]} (win {win[-1]:.3f})",
            (steps[-1], win[-1]),
            textcoords="offset points", xytext=(-12, -40), ha="right",
            fontsize=9, color=ACCENT,
            arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.0))
    # Manual annotations requested on the command line.
    by_step = {r["step"]: r for r in rows}
    for s in annotate_steps:
        if s in by_step and s != steps[peak_i]:
            r = by_step[s]
            ax_top.annotate(
                f"step {s}\nwin {r['a_win_rate']:.3f}",
                (s, r["a_win_rate"]),
                textcoords="offset points", xytext=(0, -38), ha="center",
                fontsize=9, color=ACCENT,
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.0))

    # --- Panel 2: mean score, agent vs heuristic, with the margin shaded ----
    ax_bot.fill_between(steps, b_score, a_score,
                        where=[a >= b for a, b in zip(a_score, b_score)],
                        color=AGENT, alpha=0.10, linewidth=0)
    ax_bot.plot(steps, a_score, "o-", color=AGENT, linewidth=2, markersize=6,
                label=f"{agent_label} mean score")
    ax_bot.plot(steps, b_score, "s--", color=HEUR, linewidth=1.8, markersize=5,
                label=f"{opponent} mean score")
    ax_bot.set_xlabel("Training step")
    ax_bot.set_ylabel("Mean score per game")
    ax_bot.legend(loc="lower right", frameon=False)
    ax_bot.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax_bot.set_axisbelow(True)
    for spine in ("top", "right"):
        ax_bot.spines[spine].set_visible(False)
    # Label the margin at the peak checkpoint.
    margin_peak = a_score[peak_i] - b_score[peak_i]
    ax_bot.annotate(
        f"margin {margin_peak:+.1f}",
        (steps[peak_i], (a_score[peak_i] + b_score[peak_i]) / 2),
        textcoords="offset points", xytext=(8, 0), ha="left", va="center",
        fontsize=9, color="#374151")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint-root", nargs="+", default=["checkpoints/colab_run"],
                   help="one or more directories of step_* checkpoint subdirs; "
                        "multiple roots are merged into a single curve by step "
                        "(handy when a run is split across folders)")
    p.add_argument("--agent", choices=["greedy", "mcts"], default="greedy",
                   help="policy to evaluate per checkpoint (greedy is fastest)")
    p.add_argument("--opponent", choices=["heuristic", "random"],
                   default="heuristic", help="fixed baseline to play against")
    p.add_argument("--num-games", type=int, default=256,
                   help="games per checkpoint (512 matches the headline eval)")
    p.add_argument("--sims", type=int, default=32,
                   help="MCTS simulations per move (only used when --agent mcts)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--csv", default="plots/winrate_vs_step.csv",
                   help="output CSV path (also the input for --plot-only)")
    p.add_argument("--out", default="plots/winrate_vs_step.png",
                   help="output figure path")
    p.add_argument("--annotate-step", type=int, action="append", default=[],
                   help="extra step(s) to annotate on the win-rate panel; these "
                        "are always kept even when --every subsamples")
    p.add_argument("--every", type=int, default=1,
                   help="keep only every Nth checkpoint (first, last, and "
                        "--annotate-step are always kept). Use a dense pass on "
                        "the run with the peak and a sparse pass elsewhere.")
    p.add_argument("--dry-run", action="store_true",
                   help="print the evaluate commands without running them")
    p.add_argument("--plot-only", action="store_true",
                   help="skip evaluation and re-plot from an existing --csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.plot_only:
        rows = read_csv(args.csv)
        if not rows:
            raise SystemExit(f"no rows in {args.csv}")
        plot_curve(rows, args.out, args.opponent, args.annotate_step)
        return

    rows = run_sweep(args)
    if args.dry_run:
        return
    if not rows:
        raise SystemExit("no checkpoints evaluated successfully; nothing to plot")
    plot_curve(rows, args.out, args.opponent, args.annotate_step)


if __name__ == "__main__":
    main()
