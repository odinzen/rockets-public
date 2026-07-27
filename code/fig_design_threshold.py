#!/usr/bin/env python3
"""Figures 4 and 5 on the sourced thermochemistry.

Figure 4 becomes the paper's central figure. It is no longer a "wide usable window"
map; it carries the two composition thresholds, because the equiatomic scale clears one
throat and misses the other.

Figure 5 propagates the enthalpy uncertainties as ADDITIVE kJ/mol shifts in the values
the sources actually quote, not a blanket fractional spread. Moving off the vendor
anchors shrank this band from about a factor of three to under a factor of 1.5.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sourced_volatility import P, T_RS25, T_F1, T_EDGE, p_hfzr

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
LIMIT = 100.0
COEF = {"RS-25": 2.017, "F-1": 1.708}
# quoted uncertainties on each reaction enthalpy, kJ/mol
UNC = {"Hf": 10.1, "Zr": 4.0, "Ta": 8.0, "Nb": 12.0}

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.linewidth": 1.1,
                     "axes.edgecolor": "black", "axes.facecolor": "white",
                     "figure.facecolor": "white", "savefig.facecolor": "white"})


def boxed(ax):
    for s in ax.spines.values():
        s.set_color("black")
        s.set_linewidth(1.1)
    ax.tick_params(direction="out", length=3.5, color="black")


def threshold(T, coef, limit=LIMIT):
    """Minimum hafnium fraction on the (Hf,Zr) sublattice that clears the limit.

    Recession is now coupled through the shared oxygen potential (p_hfzr), so the curve
    is not linear in x and the threshold is a root rather than a closed form. Bisect on
    the monotone recession-vs-x(Hf).
    """
    lo, hi = 0.0, 1.0
    if coef * p_hfzr(T, hi) > limit:
        return None
    if coef * p_hfzr(T, lo) <= limit:
        return 0.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if coef * p_hfzr(T, mid) > limit:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------- Figure 4, the design threshold map ----------------
x = np.linspace(0.0, 1.0, 300)


def curve(T, coef):
    return coef * np.array([p_hfzr(T, xi) for xi in x])


rs25, f1 = curve(T_RS25, COEF["RS-25"]), curve(T_F1, COEF["F-1"])
x_rs25, x_f1 = threshold(T_RS25, COEF["RS-25"]), threshold(T_F1, COEF["F-1"])

fig, ax = plt.subplots(figsize=(8.4, 5.2))
fig.subplots_adjust(top=0.86, bottom=0.20, left=0.13, right=0.96)
ax.axhspan(1e1, LIMIT, facecolor="0.90", edgecolor="none", zorder=0)
ax.plot(x, rs25, color="0.0", lw=2.0, ls=(0, (6, 3)), label="RS-25 throat, 2975 °C")
ax.plot(x, f1, color="0.0", lw=1.8, ls=(0, (1, 1.2)), label="F-1 throat, 2870 °C")
ax.axhline(LIMIT, color="0.35", lw=1.2, ls=":")
# keep annotations off both curves: the limit label sits inside the shaded band on the
# left where neither curve reaches, not on the line itself
ax.text(0.025, LIMIT * 0.72, "100 µm/h practical limit", fontsize=7.8, color="0.25")
ax.text(0.025, 13.5, "tolerable window", fontsize=8.0, color="0.35", style="italic")

for xt, lab, dy in ((x_f1, "F-1", 1.9), (x_rs25, "RS-25", 4.6)):
    ax.plot([xt, xt], [1e1, LIMIT], color="0.15", lw=1.3, ls="-", alpha=0.85)
    ax.plot([xt], [LIMIT], "o", ms=6, mfc="white", mec="black", mew=1.4, zorder=6)
    # RS-25 sits above the limit line to its right, where both curves have already
    # dropped below; F-1 sits below and right, clear of the RS-25 curve above it
    up = lab == "RS-25"
    ax.annotate(f"{lab}\nx(Hf) = {xt:.2f}", (xt, LIMIT), textcoords="offset points",
                xytext=(7, 14 if up else -12), fontsize=8.0, color="0.1",
                ha="left", va="bottom" if up else "top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="0.6", alpha=0.95))

ax.set_yscale("log")
ax.set_xlim(0, 1)
ax.set_ylim(1e1, 1e3)
ax.set_xlabel("Hafnium fraction on the (Hf,Zr)C metal sublattice, x(Hf)", fontsize=9.8)
ax.set_ylabel("Throat recession velocity, µm/h", fontsize=9.8)
ax.set_title("Hafnium Requirement Set By The Propellant", fontsize=11.0,
             fontweight="bold", pad=9)
boxed(ax)
leg = ax.legend(loc="upper right", fontsize=8.2, frameon=True, edgecolor="black")
leg.get_frame().set_linewidth(1.0)
fig.savefig(os.path.join(OUT, "fig4_design_map.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig4_design_map.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"[written] fig4_design_map  thresholds RS-25 {x_rs25:.2f}, F-1 {x_f1:.2f}")

# ---------------- Figure 5, uncertainty on the sourced enthalpies ----------------
CAND = {"C2_HfZr": ["Hf", "Zr"], "C3_HfZrTa": ["Hf", "Zr", "Ta"],
        "C3_HfZrNb": ["Hf", "Zr", "Nb"], "C4_noTi": ["Hf", "Zr", "Ta", "Nb"]}
names = list(CAND)


def _scale_p(metals, T, shift=0.0):
    """Metal pressure over an equiatomic scale: hafnia and zirconia coupled through the
    shared oxygen potential, the pentoxide and niobia added at their sublattice fraction.
    `shift` applies the per-oxide enthalpy uncertainty (+ raises enthalpy, lowers pressure).
    """
    f = 1.0 / len(metals)
    hz = (1.0 if "Hf" in metals else 0.0) + (1.0 if "Zr" in metals else 0.0)
    p = 0.0
    if hz > 0:
        x_hf = (1.0 if "Hf" in metals else 0.0) / hz
        p += p_hfzr(T, x_hf, shift * UNC["Hf"], shift * UNC["Zr"]) * (hz * f)
    if "Ta" in metals:
        p += P["Ta"](T, shift * UNC["Ta"]) * f
    if "Nb" in metals:
        p += P["Nb"](T, shift * UNC["Nb"]) * f
    return p


def band(metals, T, coef):
    base = coef * _scale_p(metals, T, 0.0)
    lo = coef * _scale_p(metals, T, +1.0)
    hi = coef * _scale_p(metals, T, -1.0)
    return base, min(lo, hi), max(lo, hi)


vals = [band(CAND[n], T_RS25, COEF["RS-25"]) for n in names]
cen = np.array([v[0] for v in vals])
lo = np.array([v[1] for v in vals])
hi = np.array([v[2] for v in vals])

fig, ax = plt.subplots(figsize=(8.2, 4.4))
fig.subplots_adjust(top=0.84, bottom=0.19, left=0.20, right=0.96)
y = np.arange(len(names))
ax.barh(y, cen, xerr=[cen - lo, hi - cen], color="0.62", edgecolor="black",
        linewidth=1.0, hatch="//", height=0.6,
        error_kw=dict(ecolor="black", elinewidth=1.2, capsize=3.5))
ax.axvline(LIMIT, color="0.3", lw=1.2, ls=":")
# below the lowest bar, not above the highest: at len(names) the label sat outside the
# axes and printed on top of the title
ax.set_ylim(-0.85, len(names) - 0.5)
ax.text(LIMIT * 1.15, -0.62, "100 µm/h limit", fontsize=8.0, color="0.25")
ax.set_xscale("log")
ax.set_yticks(y)
ax.set_yticklabels(names, fontsize=9.0)
ax.set_xlim(10, 1e4)
ax.set_xlabel("RS-25 throat recession, µm/h (log scale)", fontsize=9.5)
ax.set_title("Ranking Robustness To The Sourced Enthalpy Uncertainties",
             fontsize=10.5, fontweight="bold", pad=9)
boxed(ax)
fig.savefig(os.path.join(OUT, "fig5_uncertainty.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig5_uncertainty.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("[written] fig5_uncertainty")
for n, (b, l, h) in zip(names, vals):
    print(f"    {n:11} {b:8.0f} um/h   band {l:.0f} to {h:.0f}   span x{h/l:.2f}")
