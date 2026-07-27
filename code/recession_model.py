#!/usr/bin/env python3
"""
Odinzen engine :: dual-regime recession rate from equilibrium volatility.

Regime 1, leading edge, free molecular (Hertz Knudsen Langmuir):
    J_i = alpha_i * p_i / sqrt(2 pi M_i R T)
Regime 2, nozzle throat, diffusion limited boundary layer:
    Real throat transport from throat_transport.py (RS-25 and F-1).

Run:  python recession_model.py
Reads oxide_vaporisation_results.json if present. Prints absolute recession
velocities. If matplotlib is installed it also redraws fig3_recession_curve.png;
if not, it prints the numbers and a one line note (install with: pip install
matplotlib  or  conda install -c conda-forge matplotlib).
"""
from __future__ import annotations
import json, math, os

R = 8.314462618
UM_PER_HR = 1e6 * 3600.0          # 1 m/s -> um/h

# real throat transport for the two engines
try:
    from throat_transport import ENGINES, throat_state
    THROATS = {name: throat_state(eng) for name, eng in ENGINES.items()}
except Exception:
    THROATS = {"RS-25": {"v_per_Pa": 2.017, "Tt": 3248.0},
               "F-1":   {"v_per_Pa": 1.708, "Tt": 3143.0}}

ALPHA_LIT = {
    "SiO2 -> SiO/SiO2/O": (0.002, 0.006, "Costa & Jacobson 2017, 1748 to 1963 K"),
    "MgO (100)":          (0.06, 0.15, "Costa & Jacobson 2017, 0.107 +/- 0.047"),
    "SiO":                (0.37, 0.68, "Arrhenius alpha0 = 0.52, 1275 to 1525 K"),
    "refractory band":    (1e-4, 1e-1, "free surface 1 to 4 orders below equilibrium"),
}
ALPHA_REFRACTORY = (1e-2, 1e-1)
ALPHA_BOUND = 1.0

RHO_CERAMIC = 9.5e3
M_CERAMIC   = 0.147
VM_PER_METAL = M_CERAMIC / RHO_CERAMIC
M_VOLATILE  = 0.150
NU_METAL    = 1.0
T_LEADING   = 2200.0


def recession_free_molecular(p_Pa, alpha, T=T_LEADING, M=M_VOLATILE):
    J = alpha * p_Pa / math.sqrt(2.0 * math.pi * M * R * T)
    return NU_METAL * J * VM_PER_METAL * UM_PER_HR     # um/h


def recession_throat(p_Pa, engine):
    return THROATS[engine]["v_per_Pa"] * p_Pa          # um/h (alpha independent)


def report_point(p_Pa, label):
    lo, hi = ALPHA_REFRACTORY
    print(f"\n{label}: equilibrium volatile metal pressure = {p_Pa:.3g} Pa")
    print(f"  leading edge, alpha = 1 (upper bound) : "
          f"{recession_free_molecular(p_Pa, ALPHA_BOUND):.3g} um/h")
    print(f"  leading edge, alpha = {lo} to {hi}     : "
          f"{recession_free_molecular(p_Pa, lo):.3g} to "
          f"{recession_free_molecular(p_Pa, hi):.3g} um/h")
    for eng in THROATS:
        print(f"  throat {eng:6} diffusion limited      : "
              f"{recession_throat(p_Pa, eng):.3g} um/h")


def load_reaktoro():
    if not os.path.exists("oxide_vaporisation_results.json"):
        return None
    try:
        d = json.load(open("oxide_vaporisation_results.json"))
        out = {}
        for env, res in d.get("results", {}).items():
            for cand, r in res.items():
                p = r.get("p_metal_volatile_Pa")
                if isinstance(p, (int, float)):
                    out[(env, cand)] = float(p)
        return out or None
    except Exception:
        return None


def make_figure(points=None):
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("\n[note] matplotlib not installed, skipping the figure. "
              "Install it with:  pip install matplotlib  (or conda install -c conda-forge matplotlib)")
        return

    p = np.logspace(-6, 3, 220)
    v_bound = np.array([recession_free_molecular(x, ALPHA_BOUND) for x in p])
    v_lo = np.array([recession_free_molecular(x, ALPHA_REFRACTORY[0]) for x in p])
    v_hi = np.array([recession_free_molecular(x, ALPHA_REFRACTORY[1]) for x in p])

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.linewidth": 1.1,
                         "axes.edgecolor": "black", "axes.facecolor": "white",
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    fig.subplots_adjust(top=0.80, bottom=0.20, left=0.13, right=0.96)

    ax.fill_between(p, v_lo, v_hi, facecolor="0.78", edgecolor="black",
                    linewidth=0.8, hatch="//", label="leading edge, literature alpha 0.01 to 0.1")
    ax.plot(p, v_bound, color="black", lw=1.9, ls="-",
            label="leading edge, alpha = 1, upper bound")
    throat_styles = {"RS-25": (0, (6, 3)), "F-1": (0, (1, 1.2))}
    for eng in THROATS:
        v = np.array([recession_throat(x, eng) for x in p])
        ax.plot(p, v, color="0.0", lw=1.6, ls=throat_styles.get(eng, "--"),
                label=f"throat {eng}, diffusion limited, real transport")
    ax.axhline(100.0, color="0.4", lw=1.0, ls=":")
    ax.text(1.2e-6, 130, "100 um/h, practical service limit", fontsize=7.6, color="0.3")

    if points:
        for (lab, p_pt) in points:
            if p_pt > 0:
                ax.axvline(p_pt, color="0.55", lw=0.8, ls=":")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1e-6, 1e3); ax.set_ylim(1e-6, 1e5)
    ax.set_xlabel("Equilibrium volatile metal partial pressure, Pa", fontsize=9.5)
    ax.set_ylabel("Surface recession velocity, µm/h", fontsize=9.5)
    for s in ax.spines.values():
        s.set_color("black"); s.set_linewidth(1.1)
    ax.tick_params(direction="out", length=3.5, color="black")
    ax.set_title("RECESSION VELOCITY VERSUS EQUILIBRIUM VOLATILITY, TWO REGIMES",
                 fontsize=10.5, fontweight="bold", pad=8)
    leg = ax.legend(loc="lower right", fontsize=7.6, frameon=True, edgecolor="black")
    leg.get_frame().set_linewidth(1.0)
    fig.suptitle("Thermochemical Recession vs. Equilibrium Oxide Volatility: RS-25 and F-1 Throats",
                 fontsize=11.5, fontweight="bold", y=0.95)
    fig.text(0.5, 0.075,
             "Throat lines use real engine transport, Chapman Enskog diffusivity and a Bartz type Sherwood number. "
             "Vertical guides mark the equilibrium volatile pressures from measured oxide vaporisation (Raoult sum).",
             ha="center", fontsize=7.0, style="italic", color="0.25")
    fig.text(0.5, 0.025,
             "Leading edge: Hertz Knudsen, alpha from Costa & Jacobson 2017 and the refractory band. "
             "Throats: RS-25 and F-1 published parameters, isentropic M=1 state, throat_transport.py.",
             ha="center", fontsize=6.5, color="0.4")
    fig.savefig("fig3_recession_curve.png", dpi=200)
    plt.close(fig)
    print("\n[written] fig3_recession_curve.png")


def main():
    print("=" * 84)
    print("Evaporation coefficients (free surface, Langmuir) and real throat transport")
    print("=" * 84)
    for k, (lo, hi, src) in ALPHA_LIT.items():
        print(f"  {k:22} alpha = {lo:g} to {hi:g}   [{src}]")
    print(f"  HfO2, ZrO2 working band : alpha = {ALPHA_REFRACTORY[0]} to "
          f"{ALPHA_REFRACTORY[1]}; bound = {ALPHA_BOUND}")
    for eng, s in THROATS.items():
        print(f"  throat {eng:6} coefficient : {s['v_per_Pa']:.3f} um/h per Pa "
              f"(T_throat {s.get('Tt', float('nan')):.0f} K)")
    print("=" * 84)

    rkt = load_reaktoro()
    points = None
    if rkt:
        print("\nUsing equilibrium pressures from oxide_vaporisation_results.json:")
        points = []
        for (env, cand), p in sorted(rkt.items()):
            report_point(p, f"{env} :: {cand}")
            points.append((f"{env}:{cand}", p))
    else:
        print("\nNo Reaktoro JSON found. Illustrative anchor points; rerun after "
              "the Reaktoro step to use real pressures.")
        for p_demo, lab in [(1e-2, "illustrative low"),
                            (1.0, "illustrative mid"),
                            (1e2, "illustrative high")]:
            report_point(p_demo, lab)

    make_figure(points)


if __name__ == "__main__":
    main()
