import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sourced_volatility import p_hfzr, p_ta, T_RS25
from sourced_volatility import P as PVOL

R = 8.314462618
UM_PER_HR = 1e6 * 3600.0
THROAT_COEF = 2.017   # RS-25
T_THROAT = T_RS25     # RS-25 throat

# pure-component oxide vapor pressures at the RS-25 throat, Pa
pTa, pNb = p_ta(T_THROAT), PVOL["Nb"](T_THROAT)

def rec(metals, gamma):
    # hafnia and zirconia are coupled through the shared oxygen potential; the pentoxide
    # and niobia carry an activity coefficient gamma (a_i = gamma * x_i), the quantity
    # swept here. The recommended (Hf,Zr) scale has no pentoxide and is gamma-independent.
    x = 1.0 / len(metals)
    hz = (1.0 if "Hf" in metals else 0.0) + (1.0 if "Zr" in metals else 0.0)
    p = 0.0
    if hz > 0:
        x_hf = (1.0 if "Hf" in metals else 0.0) / hz
        p += p_hfzr(T_THROAT, x_hf) * (hz * x)
    if "Ta" in metals:
        p += gamma * x * pTa
    if "Nb" in metals:
        p += gamma * x * pNb
    return THROAT_COEF * p

gam = np.logspace(-2, 0, 100)
rNb = np.array([rec(["Hf","Zr","Nb"], g) for g in gam])
rTa = np.array([rec(["Hf","Zr","Ta"], g) for g in gam])
rC4 = np.array([rec(["Hf","Zr","Ta","Nb"], g) for g in gam])
rC2 = rec(["Hf","Zr"], 1.0)   # gamma-independent (no pentoxide)

plt.rcParams.update({"font.family":"DejaVu Sans","axes.linewidth":1.1,"axes.edgecolor":"black",
                     "axes.facecolor":"white","figure.facecolor":"white","savefig.facecolor":"white"})
fig, ax = plt.subplots(figsize=(8.0,5.0))
fig.subplots_adjust(top=0.83, bottom=0.18, left=0.13, right=0.96)
ax.axhspan(1.0, 100.0, facecolor="0.90", edgecolor="none", zorder=0)
ax.plot(gam, rNb, color="0.0", lw=1.9, ls="-", label="C3_HfZrNb (niobium)")
ax.plot(gam, rC4, color="0.0", lw=1.6, ls=(0,(5,2)), label="C4_noTi (Ta+Nb)")
ax.plot(gam, rTa, color="0.35", lw=1.9, ls=(0,(1,1.2)), label="C3_HfZrTa (tantalum)")
ax.axhline(rC2, color="0.55", lw=1.5, ls="-.", label="C2_HfZr ($HfO_2$-$ZrO_2$ near ideal)")
ax.axhline(100.0, color="0.35", lw=1.1, ls=":")
# both annotations sit inside the shaded band, which no curve enters, so neither can
# collide with a line
ax.text(0.14, 66, "100 µm/h service limit", fontsize=7.4, color="0.3")
ax.text(0.012, 30, "tolerable window, unreached at any activity",
        fontsize=7.6, style="italic", color="0.3")
# y floor at 20 rather than 1e-2: on the sourced thermochemistry every candidate
# converges on the hafnia-zirconia floor near 163 µm/h as the pentoxide contribution
# vanishes, so nothing reaches the band and the old three decades below it were empty
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1e-2,1); ax.set_ylim(20,1e4)
ax.set_xlabel("Pentoxide activity coefficient in the refractory scale, γ ($Ta_2O_5$, $Nb_2O_5$)", fontsize=9.0)
ax.set_ylabel("RS-25 throat recession, µm/h", fontsize=9.5)
ax.set_title("Ranking Robust to Non-Ideal Scale Mixing", fontsize=10.5, fontweight="bold", pad=8)
for s in ax.spines.values(): s.set_color("black"); s.set_linewidth(1.1)
ax.tick_params(direction="out", length=3.5, color="black")
leg=ax.legend(loc="upper left", fontsize=8.0, frameon=True, edgecolor="black"); leg.get_frame().set_linewidth(1.0)
import os; _fd=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","figures"); fig.savefig(os.path.join(_fd,"fig6_activity_robustness.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(_fd,"fig6_activity_robustness.png"), dpi=200, bbox_inches="tight"); plt.close(fig)
print("[written] fig6_activity_robustness.png")
print(f"pentoxide/niobia p at 3248 K (Pa): Ta={pTa:.1f} Nb={pNb:.0f}")
print(f"C2_HfZr coupled (gamma-independent): {rC2:.1f} um/h")
for g in [1.0,0.1,0.02,0.01]:
    print(f"gamma={g:5}: Nb-cand {rec(['Hf','Zr','Nb'],g):7.1f}  Ta-cand {rec(['Hf','Zr','Ta'],g):6.1f} um/h")
