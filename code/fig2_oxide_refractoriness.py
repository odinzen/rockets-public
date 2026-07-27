"""Figure 2 - oxide scale melting points against the two service windows.
Reconstructed as vector PDF. Melting points are literature values (CRC Handbook;
Fahrenholtz & Hilmas UHTC reviews); no fitted quantity here.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "figures")

# literature melting points, degC (JECS convention); the axis and both service lines
# follow. Converted from the tabulated K values (HfO2 3031, ZrO2 2988, Ta2O5 2145,
# TiO2 2116, SiO2 1986, Nb2O5 1785, B2O3 723) by subtracting 273.15.
oxides = [
    (r"$HfO_2$",  2758),
    (r"$ZrO_2$",  2715),
    (r"$Ta_2O_5$", 1872),
    (r"$TiO_2$",  1843),
    (r"$SiO_2$",  1713),
    (r"$Nb_2O_5$", 1512),
    (r"$B_2O_3$",  450),
]
# The throat line is the RS-25 gas static temperature computed in throat_transport.py,
# not a round number. At 2975 degC every candidate oxide is molten, which is the honest
# picture: throat life is set by volatility, not by which scale stays solid.
THROAT, EDGE = 2975.0, 1927.0

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.linewidth": 1.1,
                     "axes.edgecolor": "black", "axes.facecolor": "white",
                     "figure.facecolor": "white", "savefig.facecolor": "white"})
fig, ax = plt.subplots(figsize=(9.0, 5.2))
fig.subplots_adjust(top=0.80, bottom=0.16, left=0.10, right=0.97)

labels = [o[0] for o in oxides]
vals = [o[1] for o in oxides]
x = range(len(oxides))
for i, (lab, v) in enumerate(oxides):
    shade = "0.80" if lab == r"$B_2O_3$" else "0.55"
    ax.bar(i, v, width=0.62, color=shade, edgecolor="black", linewidth=1.1, hatch="///", zorder=3)

ax.axhline(THROAT, color="0.0", lw=1.8, ls="-", zorder=4)
ax.axhline(EDGE, color="0.0", lw=1.8, ls=(0, (7, 4)), zorder=4)
ax.text(len(oxides) - 0.5, THROAT + 40, "RS-25 throat gas, 2975 °C", ha="right", va="bottom",
        fontsize=9.0, fontweight="bold")
ax.text(len(oxides) - 0.5, EDGE + 40, "leading edge service, 1927 °C", ha="right", va="bottom",
        fontsize=9.0, fontweight="bold")

ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=13)
ax.set_ylim(0, 3200); ax.set_ylabel("Oxide melting temperature, °C", fontsize=10.5)
ax.set_title("Oxide Scale Refractoriness Versus Service Temperature",
             fontsize=11.5, fontweight="bold", pad=10)
for s in ax.spines.values():
    s.set_color("black"); s.set_linewidth(1.1)
ax.tick_params(direction="out", length=3.5, color="black")

fig.savefig(os.path.join(OUT, "fig2_oxide_refractoriness.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig2_oxide_refractoriness.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("[written] fig2_oxide_refractoriness.pdf + .png")
