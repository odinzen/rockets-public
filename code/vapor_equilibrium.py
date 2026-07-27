#!/usr/bin/env python3
"""
Odinzen engine :: volatile metal pressures over the oxide scale, all four
metals on measured vaporisation data, niobium measured in BOTH environments.

Niobium uses the phase that is actually stable in each environment:
  reducing throat  -> NbO2(l), NbO2(g) by the THIRD LAW from the JANAF free
                      energy functions (Kamegashira et al. 1981, Tables 5 and 8),
                      x1.4 for the measured NbO(g) channel. Confirmed against
                      their measured-pressure fit to within 6 percent.
  oxidising air    -> Nb2O5(l), NbO2(g) measured by Matsui & Naito 1983
                      log p(NbO2,g)/Pa = -0.889e4/T + 3.700 (1800-1988 K);
                      NbO and O2 were below detection, so NbO2(g) is the metal
                      bearing pressure. Leading edge value extrapolates to 2200 K.
The refractory dioxides and tantala use Clausius Clapeyron anchored to their
10^-4 Torr temperature. Raoult mixing over the scale.

Refs: Kamegashira, Matsui, Harada, Naito, J. Nucl. Mater. 101 (1981) 207;
      Matsui & Naito, Int. J. Mass Spectrom. Ion Phys. 47 (1983) 253.
"""
import json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from oxide_anchors import METAL_OXIDE, p_oxide, provenance_rows
from sourced_volatility import p_hfzr

R = 8.314462618
UM_PER_HR = 1e6 * 3600.0

THROAT_COEF = {"RS-25": 2.017, "F-1": 1.708}
M_VOLATILE = 0.150
VM_PER_METAL = 0.147 / 9.5e3
ALPHA_BOUND = 1.0
ALPHA_BAND = (1e-2, 1e-1)

CANDIDATES = {
    "C2_HfZr":   ["Hf", "Zr"],
    "C3_HfZrTa": ["Hf", "Zr", "Ta"],
    "C3_HfZrNb": ["Hf", "Zr", "Nb"],
    "C4_noTi":   ["Hf", "Zr", "Ta", "Nb"],
}
ENVIRONMENTS = {
    "Target1a_RS25_throat": dict(T=3248.0, kind="throat", engine="RS-25", oxidising=False, label="RS-25 throat (LOX/LH2, M=1)"),
    "Target1b_F1_throat":   dict(T=3143.0, kind="throat", engine="F-1",  oxidising=False, label="F-1 throat (LOX/RP-1, M=1)"),
    "Target2_leadingedge":  dict(T=2200.0, kind="leading_edge",          oxidising=True,  label="Hypersonic leading edge (air, dry)"),
}

def p_metal(metals, T, oxidising):
    """Metal-bearing pressure over an equiatomic mixed scale.

    Hafnia and zirconia share one oxygen potential (p_hfzr), not summed as independent
    oxides; the pentoxide and niobia add at their sublattice fraction. In the oxidising
    leading-edge branch the shared-oxygen coupling is negligible (pressures are ~1e-4 Pa),
    so the simple sum is retained there.
    """
    x = 1.0 / len(metals)
    if oxidising:
        return sum(x * p_oxide(METAL_OXIDE[m], T, oxidising) for m in metals)
    hz = (1.0 if "Hf" in metals else 0.0) + (1.0 if "Zr" in metals else 0.0)
    p = 0.0
    if hz > 0:
        x_hf = (1.0 if "Hf" in metals else 0.0) / hz
        p += p_hfzr(T, x_hf) * (hz * x)
    for m in metals:
        if m not in ("Hf", "Zr"):
            p += x * p_oxide(METAL_OXIDE[m], T, oxidising)
    return p

def rec_le(p, alpha, T=2200.0):
    J = alpha * p / math.sqrt(2.0 * math.pi * M_VOLATILE * R * T)
    return J * VM_PER_METAL * UM_PER_HR

def rec_throat(p, engine):
    return THROAT_COEF[engine] * p

print("=" * 96)
print("OXIDE VAPORISATION SOURCES")
for ox, prov, _src_type in provenance_rows():
    print(f"  {ox:6} [{prov}]")
print("=" * 96)

payload = {"environments": {k: v["label"] for k, v in ENVIRONMENTS.items()}, "results": {},
           "method": "measured oxide vaporisation; Nb measured both phases (Kamegashira 1981 throat, Matsui-Naito 1983 air)"}
for envkey, env in ENVIRONMENTS.items():
    T = env["T"]; ox = env["oxidising"]
    print(f"\n### {env['label']}  |  T={T:.0f} K  |  {'oxidising (Nb2O5)' if ox else 'reducing (NbO2)'}")
    rows = []
    for name, metals in CANDIDATES.items():
        p = p_metal(metals, T, ox)
        scale = "+".join(METAL_OXIDE[m] for m in metals)
        if env["kind"] == "throat":
            rec = rec_throat(p, env["engine"]); rs = f"{rec:.3g} um/h"
        else:
            rb = rec_le(p, ALPHA_BOUND); rl = rec_le(p, ALPHA_BAND[0]); rh = rec_le(p, ALPHA_BAND[1])
            rec = rb; rs = f"{rb:.2g} bound / {rl:.2g} to {rh:.2g} band um/h"
        print(f"  {name:10} {scale:24} p={p:.3e} Pa   recession {rs}")
        rows.append((name, p, rec))
        payload["results"].setdefault(envkey, {})[name] = dict(p_metal_volatile_Pa=p, scale=scale, recession_um_per_hr=rec)
    rows.sort(key=lambda r: r[1])
    print("  RANK (least volatile first): " + " > ".join(r[0] for r in rows))

print("\n" + "-" * 96)
print("Hafnium enrichment lever at the RS-25 throat (3248 K):")
lever = {}
for xHf, lab in [(0.0, "pure ZrC"), (0.5, "(Hf,Zr)C 50/50"), (1.0, "pure HfC")]:
    pv = p_hfzr(3248.0, xHf)   # coupled shared-oxygen solve, not a linear Raoult sum
    rec = rec_throat(pv, "RS-25"); lever[lab] = dict(x_Hf=xHf, p_Pa=pv, recession_um_per_hr=rec)
    print(f"  {lab:16} x_Hf={xHf:.2f}  p={pv:.3e} Pa  recession={rec:.2g} um/h")
payload["hf_enrichment_lever_RS25_throat"] = lever
print("-" * 96)

with open("oxide_vaporisation_results.json", "w") as f:
    json.dump(payload, f, indent=2)
print("\n[written] oxide_vaporisation_results.json")

# ---- fig3 ----
p = np.logspace(-6, 4, 260)
vb = np.array([rec_le(x, ALPHA_BOUND) for x in p])
vlo = np.array([rec_le(x, ALPHA_BAND[0]) for x in p]); vhi = np.array([rec_le(x, ALPHA_BAND[1]) for x in p])
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.linewidth": 1.1, "axes.edgecolor": "black",
                     "axes.facecolor": "white", "figure.facecolor": "white", "savefig.facecolor": "white"})
fig, ax = plt.subplots(figsize=(8.8, 5.4))
fig.subplots_adjust(top=0.80, bottom=0.20, left=0.13, right=0.96)
ax.fill_between(p, vlo, vhi, facecolor="0.80", edgecolor="black", linewidth=0.8, hatch="//", label="leading edge, α 0.01 to 0.1")
ax.plot(p, vb, color="black", lw=1.9, label="leading edge, α = 1, upper bound")
ax.plot(p, [rec_throat(x, "RS-25") for x in p], color="0.0", lw=1.6, ls=(0, (6, 3)), label="throat RS-25")
ax.plot(p, [rec_throat(x, "F-1") for x in p], color="0.0", lw=1.6, ls=(0, (1, 1.2)), label="throat F-1")
ax.axhline(100.0, color="0.4", lw=1.0, ls=":")
# the four throat markers sit on the RS-25 line; the three pentoxide-bearing points
# crowd the top-right within half a decade of each other, so pull every label off the
# line with a leader and stagger the vertical offsets by more than a label height.
# Equal offsets put C3_HfZrTa underneath C4_noTi and made both unreadable.
# the two leftmost labels run left into open space, the two rightmost run right, and
# each pair is separated vertically by more than a label height
LABEL_OFF = {"C2_HfZr": (-10, -16, "right"), "C3_HfZrTa": (-10, -46, "right"),
             "C3_HfZrNb": (14, -16, "left"), "C4_noTi": (14, -60, "left")}
for name, metals in CANDIDATES.items():
    px = p_metal(metals, 3248.0, False); py = rec_throat(px, "RS-25")
    ax.plot([px], [py], "o", ms=6, mfc="white", mec="black", mew=1.3, zorder=5)
    dx, dy, ha = LABEL_OFF.get(name, (12, -20, "left"))
    ax.annotate(name, (px, py), textcoords="offset points", xytext=(dx, dy),
                fontsize=6.2, color="0.1", ha=ha, va="top", zorder=6,
                arrowprops=dict(arrowstyle="-", lw=0.6, color="0.45", shrinkA=0, shrinkB=3),
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.92))
plx = p_metal(["Hf","Zr"], 2200.0, True); ply = rec_le(plx, ALPHA_BOUND)
ax.plot([plx],[ply],"s",ms=6,mfc="0.5",mec="black",mew=1.2,zorder=5)
ax.annotate("C2_HfZr leading edge",(plx,ply),textcoords="offset points",xytext=(9,-2),fontsize=6.2,color="0.1",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.92))
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1e-6,1e5); ax.set_ylim(1e-6,1e5)
ax.set_xlabel("Equilibrium volatile metal partial pressure, Pa", fontsize=9.5)
ax.set_ylabel("Surface recession velocity, µm/h", fontsize=9.5)
for s in ax.spines.values(): s.set_color("black"); s.set_linewidth(1.1)
ax.tick_params(direction="out", length=3.5, color="black")
ax.set_title("Recession From Measured Oxide Volatility, Four Survivors, RS-25 and F-1 Throats", fontsize=9.0, fontweight="bold", pad=8)
leg=ax.legend(loc="lower right", fontsize=7.4, frameon=True, edgecolor="black"); leg.get_frame().set_linewidth(1.0)
import os; _fd=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","figures"); fig.savefig(os.path.join(_fd,"fig3_recession_curve.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(_fd,"fig3_recession_curve.png"), dpi=200, bbox_inches="tight")
print("[written] fig3_recession_curve.png")
