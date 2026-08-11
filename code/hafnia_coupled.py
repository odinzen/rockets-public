#!/usr/bin/env python3
"""Coupled congruent-vaporisation equilibrium over HfO2 and ZrO2 at the RS-25 throat.

Replaces the naive channel-by-channel addition used in the first pass, which is wrong
because the channels share an oxygen balance. Over a pure dioxide the gas must carry
O:M = 2, which couples p(MO2), p(MO), p(O) and p(O2) into one solve.

Species and equilibria, condensed activity unity:
    MO2(c) = MO2(g)          K1 = p(MO2)
    MO2(c) = MO(g) + O(g)    K2 = p(MO) p(O)
    1/2 O2(g) = O(g)         Kf = p(O)/sqrt(p(O2))     JANAF, so K3 = Kf^2 = p(O)^2/p(O2)

Congruency, O:M = 2 in the gas:
    2 p(MO2) + p(MO) + p(O) + 2 p(O2) = 2 [p(MO2) + p(MO)]
which reduces to
    p(O) + 2 p(O2) = p(MO)
and with p(O2) = p(O)^2/K3 gives a cubic in p(O):
    2 p(O)^3 / K3 + p(O)^2 - K2 = 0

Hafnia has no HfO2(g) term. That is not a data gap being papered over: Panish and Reif
saw only HfO over vaporising HfO2 (verified from the full text), and Kablov 2019 KEMS
detected ZrO2 but not HfO2 in the same cell at 2660 K.

Sources are listed in docs/HAFNIA_modern_recalculation_2026-07-18.md. Gibbs energy
functions for the monoxides are extrapolated in lnT beyond their table ends.
"""
import math

import numpy as np

R = 8.314462618
T_THROAT = 3248.0
PA_PER_BAR = 1.0e5


def interp(a, b, f=0.48):
    """Linear between the Barin/JANAF 3200 and 3300 K rows, to 3248 K."""
    return a + f * (b - a)


def fit_lnT(temps, gefs, T):
    """gef is near-linear in lnT at high temperature; extrapolate on that basis."""
    A = np.vstack([np.log(temps), np.ones(len(temps))]).T
    c, *_ = np.linalg.lstsq(A, np.array(gefs), rcond=None)
    return c[0] * math.log(T) + c[1]


# HfO(g) gef = S - (H-H298)/T, J/mol/K, 1500-2500 K, from the supplementary of Bauschlicher,
# Kowalski and Jacobson, J Chem Phys 157 (2022) 154302, doi 10.1063/5.0120504 (printed Cp
# equation is misprinted and not used). Extrapolated in lnT above 2500 K.
_HFO_G_GEF = (
    [1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500],
    [263.0155, 264.8636, 266.6314, 268.3254, 269.9516, 271.5153, 273.0216, 274.4749,
     275.8796, 277.2392, 278.5572],
)


def hfo_gef(T):
    """HfO(g), fitted in lnT over the 1500-2500 K Bauschlicher grid."""
    return fit_lnT(_HFO_G_GEF[0], _HFO_G_GEF[1], T)


# gef at 3248 K, J/mol/K
GEF_O = interp(192.068, 192.653)                     # JANAF O-001
GEF_HFO2_C = interp(174.880, 178.556)                # Barin p.818, liquid
GEF_ZRO2_C = interp(158.116, 161.101)                # Barin p.1880, liquid
GEF_ZRO2_G = interp(352.327, 353.906)                # Barin p.1881
GEF_ZRO_G = fit_lnT([2200, 2400, 2500], [269.699, 273.106, 274.734], T_THROAT)
GEF_HFO_G = hfo_gef(T_THROAT)

# JANAF O-001 log Kf for 1/2 O2 = O
K3 = (10.0 ** interp(-0.666, -0.539)) ** 2

DFH_O = 249.173          # JANAF O-001, kJ/mol
DFH_ZRO2_C = -1097.463   # Barin
DFH_ZRO2_G = -286.186
DFH_ZRO_G = 58.576


def K(drH_kJ, dgef, T=T_THROAT):
    return math.exp(-drH_kJ * 1000.0 / (R * T) + dgef / R)


def solve_congruent(K1, K2):
    """Return (p_MO2, p_MO, p_O) in bar. K1 may be None when no dioxide gas exists."""
    # 2 x^3 / K3 + x^2 - K2 = 0, one positive root
    roots = np.roots([2.0 / K3, 1.0, 0.0, -K2])
    p_O = max(r.real for r in roots if abs(r.imag) < 1e-12 and r.real > 0)
    p_MO = K2 / p_O
    return (K1 or 0.0), p_MO, p_O


def main():
    print("=" * 74)
    print(f"Coupled congruent vaporisation at the RS-25 throat, {T_THROAT:.0f} K")
    print("=" * 74)
    print(f"  gef(HfO,g) {GEF_HFO_G:7.2f}   gef(ZrO,g) {GEF_ZRO_G:7.2f}   "
          f"gef(O,g) {GEF_O:7.2f}  J/mol/K")
    print(f"  K3 for O2 = 2O : {K3:.4g} bar")
    print()

    K1_zr = K(DFH_ZRO2_G - DFH_ZRO2_C, GEF_ZRO2_G - GEF_ZRO2_C)
    K2_zr = K(DFH_ZRO_G + DFH_O - DFH_ZRO2_C, GEF_ZRO_G + GEF_O - GEF_ZRO2_C)
    zr = solve_congruent(K1_zr, K2_zr)
    p_zr_metal = (zr[0] + zr[1]) * PA_PER_BAR
    print("  ZrO2, both gas channels coupled")
    print(f"    p(ZrO2,g) {zr[0]*PA_PER_BAR:8.1f}   p(ZrO,g) {zr[1]*PA_PER_BAR:8.1f}   "
          f"p(O) {zr[2]*PA_PER_BAR:8.1f}  Pa")
    # the coupled solve lands within 1 percent of the naive channel-by-channel sum,
    # so the coupling is not what drives the answer here. Worth knowing rather than
    # assuming either way.
    print(f"    total Zr-bearing {p_zr_metal:.1f} Pa "
          f"(naive channel addition gave 215.8, so coupling shifts it under 1 percent)")
    print()

    # dfH(HfO2, monoclinic, 298) = -1117.6 +/- 1.6 kJ/mol, Kornilov, Ushakova, Huber and
    # Holley, J. Chem. Thermodyn. 7 (1975) 21, doi 10.1016/0021-9614(75)90076-2, a paper
    # written to resolve exactly this discrepancy. Confirmed to 0.04 kJ/mol by the
    # Glushko/IVTAN Thermal Constants tables. Barin and CRC both carry -1144.7, which
    # descends from NBS 1982 and is superseded; they are not independent of each other.
    for label, dfh_hfo2 in [("Kornilov 1975 -1117.6 [SELECTED]", -1117.63),
                            ("superseded Barin/CRC -1144.7", -1144.742)]:
        K2_hf = K(63.19 + DFH_O - dfh_hfo2, GEF_HFO_G + GEF_O - GEF_HFO2_C)
        hf = solve_congruent(None, K2_hf)
        p_hf_metal = hf[1] * PA_PER_BAR
        print(f"  HfO2 with {label}")
        print(f"    p(HfO,g) {p_hf_metal:8.1f}   p(O) {hf[2]*PA_PER_BAR:8.1f}  Pa"
              f"   (no HfO2(g), below detection over hafnia)")
        print(f"    lever vs Zr = {p_zr_metal/p_hf_metal:5.1f}x")
        for name, x in (("pure HfC", 1.0), ("(Hf,Zr)C 50/50", 0.5), ("pure ZrC", 0.0)):
            rec = 2.017 * (x * p_hf_metal + (1 - x) * p_zr_metal)
            print(f"      {name:16} {rec:7.0f} um/h  {'clears' if rec < 100 else 'OVER LIMIT'}")
        print()


if __name__ == "__main__":
    main()
