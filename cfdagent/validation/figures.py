"""Figures for the transition-condition validation study."""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Validated categorical palette (light surface #fcfcfb):
# CVD separation PASS, normal-vision floor PASS. The aqua slot warns on
# contrast, so every series is directly labelled rather than relying on colour.
SERIES = {"pred": "#2a78d6", "tripped": "#eb6834", "free": "#1baf7a"}
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#8a8880"}
SURFACE = "#fcfcfb"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK["muted"])
        ax.spines[side].set_linewidth(0.8)
    ax.grid(True, color=INK["muted"], alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK["secondary"], labelsize=9)


def drag_polar(out_path: Path) -> Path:
    """Drag polar: prediction against both experiments."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .experimental import load_abbott, load_ladson
    from .transition_study import neuralfoil_polar

    ladson, abbott = load_ladson(), load_abbott()
    cl, cd = neuralfoil_polar(6.0e6)
    order = np.argsort(cl)

    fig, ax = plt.subplots(figsize=(6.4, 4.6), facecolor=SURFACE)
    _style(ax)

    ax.plot(cd[order], cl[order], color=SERIES["pred"], linewidth=2.0, zorder=3)
    ax.scatter(ladson.cd, ladson.cl, s=26, color=SERIES["tripped"],
               edgecolor=SURFACE, linewidth=0.8, zorder=4)
    ax.scatter(abbott.cd, abbott.cl, s=44, marker="D", color=SERIES["free"],
               edgecolor=SURFACE, linewidth=0.8, zorder=4)

    # Direct labels, placed in separate regions of the axes so they do not
    # collide. Identity never rests on colour alone.
    ax.annotate("NeuralFoil\n(models transition)",
                xy=(cd[order][-3], cl[order][-3]), xytext=(0.052, 0.90),
                textcoords="axes fraction", color=INK["primary"], fontsize=9,
                ha="left", va="top",
                arrowprops=dict(arrowstyle="-", color=INK["muted"], linewidth=0.8))
    ax.annotate("Ladson — tripped\n(turbulent from LE)",
                xy=(ladson.cd[3], ladson.cl[3]), xytext=(0.60, 0.42),
                textcoords="axes fraction", color=INK["primary"], fontsize=9,
                ha="left", va="top",
                arrowprops=dict(arrowstyle="-", color=INK["muted"], linewidth=0.8))
    ax.annotate("Abbott — free transition",
                xy=(abbott.cd[-3], abbott.cl[-3]), xytext=(0.40, 0.20),
                textcoords="axes fraction", color=INK["primary"], fontsize=9,
                ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=INK["muted"], linewidth=0.8))

    ax.set_xlabel("$C_D$", color=INK["secondary"], fontsize=10)
    ax.set_ylabel("$C_L$", color=INK["secondary"], fontsize=10)
    ax.set_title("NACA0012 drag polar, Re = 6×10⁶",
                 color=INK["primary"], fontsize=12, pad=12, loc="left")
    ax.set_xlim(0.004, 0.017)
    ax.set_ylim(-0.75, 1.55)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def error_summary(out_path: Path) -> Path:
    """Where the error lands: drag against each experiment, and lift."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .transition_study import run

    r = run()
    labels = ["Drag vs tripped\n(Ladson)", "Drag vs free\n(Abbott)", "Lift vs tripped\n(Ladson)"]
    values = [r["drag_tripped"].mean_abs_pct, r["drag_free"].mean_abs_pct,
              r["lift_tripped"].mean_abs_pct]
    colors = [SERIES["tripped"], SERIES["free"], SERIES["pred"]]

    fig, ax = plt.subplots(figsize=(6.4, 4.0), facecolor=SURFACE)
    _style(ax)
    ax.grid(True, axis="x", color=INK["muted"], alpha=0.25, linewidth=0.6)
    ax.grid(False, axis="y")

    bars = ax.barh(labels, values, color=colors, height=0.55, zorder=3)
    for bar, v in zip(bars, values):
        ax.text(v + 0.8, bar.get_y() + bar.get_height() / 2, f"{v:.1f}%",
                va="center", color=INK["primary"], fontsize=10, fontweight="medium")

    ax.set_xlabel("mean absolute error vs experiment (%)",
                  color=INK["secondary"], fontsize=10)
    ax.set_title("Same model, same airfoil, same Reynolds number",
                 color=INK["primary"], fontsize=12, pad=12, loc="left")
    ax.set_xlim(0, max(values) * 1.25)
    ax.invert_yaxis()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def main() -> None:
    out = Path("reports/figures")
    print("wrote", drag_polar(out / "drag_polar.png"))
    print("wrote", error_summary(out / "error_summary.png"))


if __name__ == "__main__":
    main()
