import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from oxide_anchors import p_oxide_for_metal

R = 8.314462618
T = 3873.0   # 3600 C, Wen et al. 2025 extreme-temperature test

metals = ["Hf","Zr","Ta","Nb"]
labels = ["Hf\n$HfO_2$","Zr\n$ZrO_2$","Ta\n$Ta_2O_5$","Nb\n$NbO_2$"]
p = [p_oxide_for_metal(m, T) for m in metals]
for m,v in zip(metals,p): print(f"{m}: p={v:.3g} Pa at {T:.0f} K")

plt.rcParams.update({"font.family":"DejaVu Sans","axes.linewidth":1.1,"axes.edgecolor":"black",
                     "axes.facecolor":"white","figure.facecolor":"white","savefig.facecolor":"white"})
fig, ax = plt.subplots(figsize=(7.4,4.8))
fig.subplots_adjust(top=0.82, bottom=0.18, left=0.13, right=0.96)
sh = ["0.45","0.45","0.30","0.30"]; ht=["//","//","xx","xx"]
x = np.arange(len(metals))
for xi,(v,s,h) in enumerate(zip(p,sh,ht)):
    ax.bar(xi, v, color=s, edgecolor="black", lw=1.1, hatch=h, width=0.62)
ax.set_yscale("log")
# backbone vs liability brackets, lifted clear of the bars so the labels never overlap them
ax.annotate("", xy=(1.4,9e4), xytext=(-0.4,9e4), arrowprops=dict(arrowstyle="-",lw=1.0,color="0.3"))
ax.text(0.5, 1.2e5, "refractory backbone", ha="center", fontsize=7.6, color="0.25", style="italic")
ax.annotate("", xy=(3.4,9e4), xytext=(1.6,9e4), arrowprops=dict(arrowstyle="-",lw=1.0,color="0.3"))
ax.text(2.5, 1.2e5, "volatile liabilities", ha="center", fontsize=7.6, color="0.25", style="italic")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9.0, fontweight="bold")
ax.set_ylabel("Equilibrium volatile metal pressure at 3600 °C, Pa", fontsize=9.5)
ax.set_ylim(10, 3e5)
ax.set_title("A Zero-Computation Descriptor Recovers the Metal Selection Rule", fontsize=9.6, fontweight="bold", pad=8)
for s in ax.spines.values(): s.set_color("black"); s.set_linewidth(1.1)
ax.tick_params(direction="out", length=3.5, color="black")
import os; _fd=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","figures"); fig.savefig(os.path.join(_fd,"fig7_design_rule.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(_fd,"fig7_design_rule.png"), dpi=200, bbox_inches="tight"); plt.close(fig)
print("[written] fig7_design_rule.png")
