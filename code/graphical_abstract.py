import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import numpy as np

plt.rcParams.update({"font.family":"DejaVu Sans","axes.linewidth":1.1,"axes.edgecolor":"black",
                     "axes.facecolor":"white","figure.facecolor":"white","savefig.facecolor":"white"})

# graphical abstract: landscape, ~13 cm wide
fig = plt.figure(figsize=(7.4, 3.5))
gs = fig.add_gridspec(1, 5, left=0.015, right=0.985, top=0.80, bottom=0.16, wspace=0.0)
axS = fig.add_subplot(gs[0, 0])     # architecture schematic
axL = fig.add_subplot(gs[0, 1:])    # recession ladder

# ---- left: graded architecture schematic ----
axS.set_xlim(0, 1); axS.set_ylim(0, 1); axS.axis("off")
axS.add_patch(Rectangle((0.18,0.18),0.64,0.46, facecolor="0.62", edgecolor="black", lw=1.2, hatch="//"))
axS.add_patch(Rectangle((0.18,0.64),0.64,0.10, facecolor="0.30", edgecolor="black", lw=1.2, hatch="xx"))
axS.text(0.5,0.41,"high-entropy\ncarbide bulk", ha="center", va="center", fontsize=6.6, color="white", fontweight="bold")
axS.text(0.5,0.69,"Hf, Zr-rich surface", ha="center", va="center", fontsize=5.8, color="white", fontweight="bold")
axS.annotate("", xy=(0.5,0.86), xytext=(0.5,0.76), arrowprops=dict(arrowstyle="-|>", lw=1.3, color="black"))
axS.text(0.5,0.93,"hot gas", ha="center", fontsize=6.4, color="0.2")
axS.text(0.5,0.07,"graded architecture", ha="center", fontsize=6.6, style="italic", color="0.25")

# ---- right: recession ladder (log x) ----
mats = [
    ("Carbon-carbon, incumbent", 2.0e5, "0.30", "xx"),
    (r"$ZrB_2$-SiC-ZrC, measured", 4.3e2, "0.55", "\\\\"),
    ("(Hf,Zr)C, this work",       265.0, "0.72", "//"),
    ("HfC-SiC, measured",         1.0e2, "0.55", "\\\\"),
    ("pure HfC",                  50.0,  "0.72", "//"),
]
y = np.arange(len(mats))[::-1]
for yi,(lab,val,sh,ht) in zip(y, mats):
    axL.barh(yi, val, color=sh, edgecolor="black", lw=1.0, hatch=ht, height=0.62)
    axL.text(val*1.4, yi, lab, va="center", fontsize=6.6, color="0.1")
axL.axvline(100.0, color="0.35", lw=1.1, ls=":")
axL.text(100, len(mats)-0.35, "100 µm/h\nservice limit", ha="center", fontsize=5.8, color="0.3")
# hafnium enrichment moves the equiatomic scale (above the limit) down to pure HfC
# (below it); the arrow crosses the service-limit line, which is the design message
axL.annotate("hafnium enrichment\ncrosses the limit", xy=(55, 0.18), xytext=(8e4, 1.15),
             fontsize=6.2, color="0.1", ha="center", va="center",
             arrowprops=dict(arrowstyle="-|>", lw=1.4, color="black",
                             connectionstyle="arc3,rad=0.28", shrinkB=4))
axL.set_xscale("log"); axL.set_xlim(1, 5e7); axL.set_ylim(-0.6, len(mats)-0.4)
axL.set_yticks([]); axL.set_xlabel("Surface recession in a rocket throat, µm/h", fontsize=7.6)
for s in axL.spines.values(): s.set_color("black"); s.set_linewidth(1.1)
axL.spines["left"].set_visible(False)
axL.tick_params(direction="out", length=3.0, color="black", labelsize=6.8)

# Design provenance for the graphical abstract. The image shipped in the repo
# root is the journal editorial-final (greyscale); this reproduces its content.
fig.suptitle("Surface oxidation, not bulk stability, determines high entropy carbide selection",
             fontsize=8.0, fontweight="bold", y=0.955)
import os
_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
# keep the deliverable free of any tool/provenance metadata (no baked strings in the file)
fig.savefig(os.path.join(_root, "graphical_abstract.pdf"), bbox_inches="tight",
            metadata={"Creator": "", "Producer": ""})
fig.savefig(os.path.join(_root, "graphical_abstract.png"), dpi=300, bbox_inches="tight",
            metadata={"Software": ""})
print("[written] graphical_abstract.png")
