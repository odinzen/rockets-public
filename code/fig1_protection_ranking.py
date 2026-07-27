"""Figure 1 - surface protection ranking of the candidates in the two service
environments, as vector PDF.

The dimensionless "surface protection index" is a composite screen output (it
combines refractory-backbone margin, molten-oxide count, and boria/silica
volatility). Its per-candidate values are carried from the established screen
result. Provenance line corrected: the pair enthalpies feeding the screen are
computed in this work by DFT, not adopted from Pak & Kvashnin.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "figures")

# surface protection index per candidate, per environment (screen output)
THROAT = {"C2_HfZr": -1.3, "C3_HfZrNb": -3.0, "C3_HfZrTa": -3.2, "C4_noTi": -4.7,
          "C5": -6.0, "B5": -20.0, "ZrB2_SiC": -21.0}
EDGE = {"C2_HfZr": 3.3, "C3_HfZrNb": 1.9, "C3_HfZrTa": 1.8, "C4_noTi": 0.3,
        "C5": -1.1, "ZrB2_SiC": -7.5, "B5": -10.3}
DIBORIDE = {"B5", "ZrB2_SiC"}


def style(name):
    if name in DIBORIDE:
        return dict(color="0.25", hatch="xxx")      # diboride, (Me)B2
    return dict(color="0.65", hatch="///")           # rocksalt carbide, (Me)C


plt.rcParams.update({"font.family": "DejaVu Sans", "axes.linewidth": 1.1,
                     "axes.edgecolor": "black", "axes.facecolor": "white",
                     "figure.facecolor": "white", "savefig.facecolor": "white"})
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.0, 6.2))
fig.subplots_adjust(top=0.80, bottom=0.24, left=0.11, right=0.97, wspace=0.30)


def panel(ax, data, title):
    order = sorted(data, key=lambda k: data[k])  # most negative at the bottom
    ys = range(len(order))
    for y, name in zip(ys, order):
        ax.barh(y, data[name], height=0.66, edgecolor="black", linewidth=1.1,
                zorder=3, **style(name))
    ax.axvline(0.0, color="0.0", lw=1.2, ls=(0, (4, 3)), zorder=4)
    ax.set_yticks(list(ys)); ax.set_yticklabels(order, fontsize=11)
    ax.set_xlabel("Surface protection index, dimensionless", fontsize=10.5)
    ax.set_title(title, fontsize=11.5, fontweight="bold", pad=8)
    for s in ax.spines.values():
        s.set_color("black"); s.set_linewidth(1.1)
    ax.tick_params(direction="out", length=3.5, color="black")


panel(axL, THROAT, "TARGET 1  |  NOZZLE THROAT")
panel(axR, EDGE, "TARGET 2  |  HYPERSONIC LEADING EDGE")

# shared legend
from matplotlib.patches import Patch
handles = [Patch(facecolor="0.65", edgecolor="black", hatch="///", label="rocksalt carbide, (Me)C"),
           Patch(facecolor="0.25", edgecolor="black", hatch="xxx", label=r"diboride, (Me)B$_2$")]
leg = fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=10.5,
                 frameon=True, edgecolor="black", bbox_to_anchor=(0.5, 0.03))
leg.get_frame().set_linewidth(1.0)

fig.suptitle("Surface Protection Ranking of Candidate UHTC Compositions, by Service Environment",
             fontsize=13.5, fontweight="bold", y=0.955)

fig.savefig(os.path.join(OUT, "fig1_protection_ranking.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig1_protection_ranking.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("[written] fig1_protection_ranking.pdf + .png")
