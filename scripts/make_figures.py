"""make_figures.py — figures for the paper (light/print). Fig. 1 (25-motif shares, ordered by
retention; fig2_motif_map), SI Fig. S9-1 (form-compliance heatmap; fig1_form_heatmap), SI Fig. S1-1
(per-system 25-motif shares).
Sources: results/ CSVs + generation logs.
Colors: validated defaults (blue #2a78d6 / orange #eb6834 poles; sequential blue ramp; neutral
grays). Run: python3 scripts/make_figures.py  → figures/*.png(.pdf)"""
from __future__ import annotations
import json, math
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "text.color": "#0b0b0b", "axes.edgecolor": "#b9b8b2",
                     "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "font.size": 10})
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY_D, GRAY_M, GRAY_L = "#52514e", "#8a8985", "#d8d7d2"

NAME = {0:"sea", 1:"flowers", 2:"family", 3:"food/animals", 4:"affect", 5:"school/child",
        6:"winter", 7:"rain", 8:"woman", 9:"old age/underground", 10:"birds/butterflies",
        11:"meta-poetic", 12:"letters/writing", 13:"interior", 14:"music", 15:"street",
        16:"cat", 17:"cosmos", 18:"wounds/drink", 19:"fruit", 20:"grandparents", 21:"mirror",
        22:"hospital/illness", 23:"apartment/wall", 24:"bus"}
SYS5 = ["GPT-5.6", "Opus-4.8", "Gemini-2.5", "Solar-Pro3", "EXAONE-4.0"]


def fmtpct(v):
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return s


SUBMISSION_NAME = {"fig2_motif_map": "figure1", "fig1_form_heatmap": "figureS9-1", "figS6_motif_by_system": "figureS1-1"}


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    # submission copy: TIFF at 600 dpi (journal requirement for colour line art), DOS-style name
    sub = SUBMISSION_NAME.get(name)
    if sub:
        fig.savefig(FIG / f"{sub}.tif", dpi=600, bbox_inches="tight", facecolor="white",
                    pil_kwargs={"compression": "tiff_lzw"})
        from PIL import Image
        im = Image.open(FIG / f"{sub}.tif")
        if im.mode == "RGBA":   # journals expect RGB without an alpha channel
            bg = Image.new("RGB", im.size, (255, 255, 255)); bg.paste(im, mask=im.split()[3])
            bg.save(FIG / f"{sub}.tif", dpi=(600, 600), compression="tiff_lzw")
    plt.close(fig)
    print(f"  {name} saved")


def fig_main2_motif_map():
    d = pd.read_csv(ROOT / "results/phase2/motif_full_distribution_balanced.csv")
    human = d["human_%"].values
    gen = d[[f"{m}_%" for m in SYS5]].values.mean(1)
    ret = gen / human
    order = np.argsort(ret)  # ordered by retention (top = highest), the comparison statistic of §2.4.2
    fig, ax = plt.subplots(figsize=(6.6, 8.2))
    for row, j in enumerate(order):
        over = ret[j] > 1
        ax.plot([human[j], gen[j]], [row, row], color=GRAY_L, lw=1.6, zorder=1)
        ax.scatter(human[j], row, s=32, color=GRAY_D, zorder=3)
        ax.scatter(gen[j], row, s=46, color=(ORANGE if over else BLUE), zorder=3)
    k_below = int((ret[order] < 1).sum())  # number of motifs below the retention = 1 boundary
    ax.axhline(k_below - 0.5, color=GRAY_M, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.annotate("retention = 1", (12.9, k_below - 0.5), xytext=(0, 4), textcoords="offset points",
                ha="right", fontsize=8.5, color=GRAY_M)
    ax.set_yticks(range(25))
    ax.set_yticklabels([f"T{j} {NAME[j]}" for j in order], fontsize=9)
    ax.set_xlabel("mean motif share (%)")
    ax.set_xlim(-0.3, 13)
    ax.grid(axis="x", color="#eceae6", lw=0.8, zorder=0)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    # direct labels: extremes only
    for j in order[-3:]:
        ax.annotate(f"{fmtpct(gen[j])}%", (gen[j], list(order).index(j)), xytext=(6, -3),
                    textcoords="offset points", fontsize=8.5, color="#0b0b0b")
    for j in order[:3]:
        ax.annotate(f"{fmtpct(gen[j])}%", (gen[j], list(order).index(j)), xytext=(6, -3),
                    textcoords="offset points", fontsize=8.5, color="#0b0b0b")
    handles = [plt.Line2D([], [], marker="o", ls="", color=GRAY_D, label="human (N = 4,642)"),
               plt.Line2D([], [], marker="o", ls="", color=ORANGE, label="generated \u2014 above human"),
               plt.Line2D([], [], marker="o", ls="", color=BLUE, label="generated \u2014 below human")]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9)
    save(fig, "fig2_motif_map")


def fig_main1_form_heatmap():
    frozen = json.loads((ROOT / "docs/prompts_frozen_2026-07-28.json").read_text())
    recs = frozen if isinstance(frozen, list) else frozen.get("prompts", frozen)
    meta = {r["prompt_id"]: r for r in recs}
    frames = {"instruction": "instruction", "completion": "completion", "persona": "persona"}
    order = [pid for f in frames for pid in sorted(meta) if meta[pid]["framing"] == f]
    STEM = {"GPT-5.6": "openai_gpt-5.6", "Gemini 2.5": "google_gemini-2.5-pro",
            "Opus 4.8": "anthropic_opus-4-8", "Solar Pro 3": "upstage_solar-pro3",
            "EXAONE 4.0": "lg_exaone", "HCX-005": "naver_hcx-005"}
    M = np.zeros((len(STEM), 12))
    for i, (name, stem) in enumerate(STEM.items()):
        raw = defaultdict(list)
        for l in open(ROOT / f"data/generated/arm_ab/{stem}.jsonl", encoding="utf-8"):
            r = json.loads(l); raw[r["prompt_id"]].append(r)
        for jx, pid in enumerate(order):
            rs = sorted(raw[pid], key=lambda r: r.get("attempt", r.get("seed_idx", 0)))
            v = a = 0                       # protocol sample: stop at the 25th valid (§2.3)
            for r in rs:
                if r.get("reason") == "error": continue   # transport failures (HTTP 4xx/429) are not completed calls: excluded
                a += 1; v += bool(r.get("is_valid"))
                if v == 25: break
            M[i, jx] = v / a if a else np.nan
    cmap = LinearSegmentedColormap.from_list("blue_seq", ["#f4f8fd", "#123f75"])
    fig, ax = plt.subplots(figsize=(9.8, 3.4))
    ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    def fmt(v):
        pct = v * 100
        return f"{pct:.0f}" if abs(pct - round(pct)) < 1e-9 else f"{pct:.2f}"

    for i in range(M.shape[0]):
        for jx in range(12):
            val = M[i, jx]
            ax.text(jx, i, fmt(val), ha="center", va="center", fontsize=7.5,
                    color=("#ffffff" if val > 0.55 else "#0b0b0b"))
    labels = [f"{pid}{'*' if meta[pid]['gloss'] else ''}" for pid in order]
    ax.set_xticks(range(12)); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_yticks(range(len(STEM))); ax.set_yticklabels(list(STEM), fontsize=9)
    # framing group separators + headers
    counts = [sum(1 for p in order if meta[p]["framing"] == f) for f in frames]
    edges = np.cumsum(counts)[:-1]
    for e in edges: ax.axvline(e - 0.5, color="#fcfcfb", lw=2.5)
    start = 0
    for f, c in zip(frames.values(), counts):
        ax.text(start + c / 2 - 0.5, -0.75, f, ha="center", fontsize=9.5, color=GRAY_D)
        start += c
    ax.set_xlabel("prompt (*: genre definition included)")
    for s in ax.spines.values(): s.set_visible(False)
    save(fig, "fig1_form_heatmap")



def fig_S6_by_system():
    """SI Fig. S1-1: per-system small multiples of the 25-motif shares (order and colours as in Fig. 1)."""
    d = pd.read_csv(ROOT / "results/phase2/motif_full_distribution_balanced.csv")
    human = d["human_%"].values
    gen5 = d[[f"{m}_%" for m in SYS5]].values.mean(1)
    order = np.argsort(gen5 / human)  # same ordering as Fig. 1: descending retention
    systems = SYS5  # the five form-compliant systems (§3)
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 16.5), sharey=True)
    for k, (sys, ax) in enumerate(zip(systems, axes.flat)):
        g = d[f"{sys}_%"].values
        for row, j in enumerate(order):
            over = g[j] > human[j]
            ax.plot([human[j], g[j]], [row, row], color=GRAY_L, lw=1.3, zorder=1)
            ax.scatter(human[j], row, s=18, color=GRAY_D, zorder=3)
            ax.scatter(g[j], row, s=28, color=(ORANGE if over else BLUE), zorder=3)
        ax.set_yticks(range(25))
        if k % 2 == 0:
            ax.set_yticklabels([f"T{j} {NAME[j]}" for j in order], fontsize=8)
        ax.set_xlim(-0.4, 17)
        ax.grid(axis="x", color="#eceae6", lw=0.8, zorder=0)
        for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
        ax.set_title(sys, fontsize=10)
        jmax = order[np.argmax([g[j] for j in order])]
        ax.annotate(f"{fmtpct(g[jmax])}%", (g[jmax], list(order).index(jmax)), xytext=(4, -3),
                    textcoords="offset points", fontsize=8, color="#0b0b0b")
    for ax in axes.flat[len(systems):]: ax.set_visible(False)
    handles = [plt.Line2D([], [], marker="o", ls="", color=GRAY_D, label="human"),
               plt.Line2D([], [], marker="o", ls="", color=ORANGE, label="system \u2014 above human"),
               plt.Line2D([], [], marker="o", ls="", color=BLUE, label="system \u2014 below human")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.01, 1, 0.99))
    save(fig, "figS6_motif_by_system")


if __name__ == "__main__":
    fig_main1_form_heatmap(); fig_main2_motif_map(); fig_S6_by_system()
    print("done ->", FIG)
