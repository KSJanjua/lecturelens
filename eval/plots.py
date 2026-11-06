"""Report plots (matplotlib, PNG). Categorical colours are used in a fixed order,
text stays in neutral ink, one y-axis per chart, and points are direct-labelled."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _style(ax, title, xlabel="", ylabel=""):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=12)
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.tick_params(colors=INK_2, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def strategy_bars(summary: list[dict], path, metrics=("rougeL", "bertscore_f1", "faithfulness")):
    """Grouped bars: one group per metric, one bar per prompting strategy (all on a 0-1 scale)."""
    labels = {"rougeL": "ROUGE-L", "bertscore_f1": "BERTScore F1", "faithfulness": "Faithfulness"}
    metrics = [m for m in metrics if any(r.get(m) is not None for r in summary)]
    if not metrics or not summary:
        return None
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    n = len(summary)
    width = 0.8 / n
    for i, row in enumerate(summary):
        xs = [m + (i - (n - 1) / 2) * width for m in range(len(metrics))]
        ys = [row.get(m) or 0 for m in metrics]
        ax.bar(xs, ys, width=width - 0.02, color=SERIES[i], label=row["strategy_label"], edgecolor=SURFACE,
               linewidth=1)
    ax.set_xticks(range(len(metrics)), [labels.get(m, m) for m in metrics])
    ax.set_ylim(0, 1)
    _style(ax, "Notes quality by prompting strategy", ylabel="Score (0-1)")
    ax.legend(frameon=False, ncol=min(4, n), loc="upper left", bbox_to_anchor=(0, -0.08), labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def quality_vs_cost(summary: list[dict], path, metric="faithfulness"):
    """One point per strategy, labelled directly (no colour legend needed)."""
    rows = [r for r in summary if r.get(metric) is not None and r.get("tokens")]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=150)
    ax.scatter([r["tokens"] for r in rows], [r[metric] for r in rows], s=70, color=SERIES[0],
               edgecolor=SURFACE, linewidth=2, zorder=3)
    for r in rows:
        ax.annotate(r["strategy_label"], (r["tokens"], r[metric]), textcoords="offset points", xytext=(8, 6),
                    color=INK, fontsize=9)
    _style(ax, f"{metric.replace('_', ' ').title()} vs cost", xlabel="Tokens per lecture",
           ylabel=metric.replace("_", " ").title())
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def diversity_vs_faithfulness(rows: list[dict], path):
    """Experiment 3: two small multiples against temperature (faithfulness, Distinct-2), one line per top-p."""
    if not rows:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), dpi=150, sharex=True)
    top_ps = sorted({r["top_p"] for r in rows})
    for ax, metric, title in ((axes[0], "faithfulness", "Faithfulness"), (axes[1], "distinct2", "Distinct-2 (diversity)")):
        for i, top_p in enumerate(top_ps):
            pts = sorted((r for r in rows if r["top_p"] == top_p), key=lambda r: r["temperature"])
            ax.plot([p["temperature"] for p in pts], [p[metric] for p in pts], color=SERIES[i], linewidth=2,
                    marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5, label=f"top-p {top_p}")
        _style(ax, title, xlabel="Temperature")
        ax.set_xticks(sorted({r["temperature"] for r in rows}))
        ax.set_ylim(0, 1.05)
    axes[0].legend(frameon=False, labelcolor=INK_2, loc="lower left")
    fig.suptitle("Sampling settings vs faithfulness and diversity", x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
