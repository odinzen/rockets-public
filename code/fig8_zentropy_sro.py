"""Figure 8 - short-range-order correction to the configurational entropy of the
candidate carbides across the service window, from the pyzentropy recursive-entropy
refinement (code/zentropy_sro_results.json, produced by _work/zentropy_lightpath.py).

The trim is the finite-cell ideal configurational entropy minus the zentropy entropy,
i.e. how far short-range order pulls the true entropy below random mixing. Greyscale
house style; caption lives in the manuscript, not on the plot.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "figures")
data = json.load(open(os.path.join(HERE, "zentropy_sro_results.json")))
T = data["temperatures_K"]

# darkest = most ordering (five-metal, Ti-bearing); lightest = near-ideal binary
STYLE = {
    "C2_HfZr":   dict(color="0.72", ls="-",  marker="o", label="(Hf,Zr)C"),
    "C3_HfZrNb": dict(color="0.58", ls="--", marker="s", label="(Hf,Zr,Nb)C"),
    "C3_HfZrTa": dict(color="0.44", ls="-.", marker="^", label="(Hf,Zr,Ta)C"),
    "C4_noTi":   dict(color="0.28", ls=":",  marker="D", label="(Hf,Zr,Ta,Nb)C"),
    "C5":        dict(color="0.05", ls="-",  marker="v", label="(Hf,Zr,Ti,Ta,Nb)C"),
}

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.linewidth": 1.1,
                     "axes.edgecolor": "black", "axes.facecolor": "white",
                     "figure.facecolor": "white", "savefig.facecolor": "white"})
fig, ax = plt.subplots(figsize=(7.6, 5.2))
fig.subplots_adjust(top=0.88, bottom=0.13, left=0.13, right=0.96)

for key in ["C5", "C4_noTi", "C3_HfZrTa", "C3_HfZrNb", "C2_HfZr"]:
    s = STYLE[key]
    y = data["candidates"][key]["SRO_trim"]
    ax.plot(T, y, color=s["color"], ls=s["ls"], marker=s["marker"], markersize=4.5,
            markevery=4, lw=1.6, label=s["label"], zorder=3)

# one percent of the smallest multi-metal ideal entropy, as an eye guide
ref = 0.01 * min(data["candidates"][k]["S_cell_ideal"] for k in STYLE if k != "C2_HfZr")
ax.axhline(ref, color="0.0", lw=1.0, ls=(0, (4, 3)), zorder=2)
ax.text(T[0] + 60, ref * 1.15, "1% of ideal S", ha="left", va="bottom", fontsize=9, color="0.0")

ax.set_yscale("log")
ax.set_xlim(T[0], T[-1])
ax.set_xlabel("Temperature (K)", fontsize=11)
ax.set_ylabel(r"Short-range-order entropy trim, $S_\mathrm{ideal}-S_\mathrm{zentropy}$ (J mol$^{-1}$ K$^{-1}$)",
              fontsize=10.5)
for sp in ax.spines.values():
    sp.set_color("black"); sp.set_linewidth(1.1)
ax.tick_params(direction="out", length=3.5, color="black")
ax.grid(True, which="major", axis="both", color="0.85", lw=0.6, zorder=0)
leg = ax.legend(loc="upper right", fontsize=9.5, frameon=True, edgecolor="black", ncol=1)
leg.get_frame().set_linewidth(1.0)

fig.suptitle("Short-Range-Order Correction to the Configurational Entropy Across the Service Window",
             fontsize=12.5, fontweight="bold", y=0.965)

os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, "fig8_zentropy_sro.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig8_zentropy_sro.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("[written] fig8_zentropy_sro.pdf + .png")
