#!/usr/bin/env python3
"""Volatile metal pressures over the oxide scales, on sourced thermochemistry.

Replaces the Clausius-Clapeyron anchors in oxide_anchors.py, which were Kurt J. Lesker
PVD vendor-datasheet temperatures paired with enthalpies that had no source at all. Every
number here traces to primary measurement or an evaluated compilation, and every
vaporisation reaction is the one the species actually follow.

Reactions, condensed activity unity:
    HfO2(c)  = HfO(g) + O(g)                 no gaseous dioxide, see below
    ZrO2(c)  = ZrO2(g)   and  = ZrO(g)+O(g)  both channels real, solved coupled
    Ta2O5(l) = 2 TaO2(g) + 1/2 O2(g)         incongruent from a melt, no gaseous pentoxide
    NbO2(l)  = NbO2(g)                       third law, Kamegashira; unchanged from before

Hafnia carries only the monoxide on evidence, not assumption. Panish and Reif saw no
higher-mass Hf species at a sensitivity floor of about 2 percent of the HfO+ signal, and
Kablov 2019 KEMS detected ZrO2 but not HfO2 in the same cell at 2660 K. Zirconia genuinely
does carry both channels, so the asymmetry is chemistry rather than missing data.

Sources
  dfH(HfO2,s)          Kornilov, Ushakova, Huber, Holley, J Chem Thermodyn 7 (1975) 21,
                       doi 10.1016/0021-9614(75)90076-2. Primary combustion calorimetry,
                       -1117.6 +/- 1.6 kJ/mol. Supersedes the NBS-1982 value that Barin
                       and CRC both carry.
  HfO(g) functions     Bauschlicher, Kowalski, Jacobson, J Chem Phys 157 (2022) 154302,
                       doi 10.1063/5.0120504, supplementary table. Its printed Cp equation
                       is misprinted and is NOT used.
  O(g), O2(g)          NIST-JANAF web tables O-001 and O-029.
  Zr and Ta oxides     Barin 1995, Thermochemical Data of Pure Substances.
  Nb                   Kamegashira 1981 (throat), Matsui and Naito 1983 (air).
  hafnia cross-check   Panish and Reif, J Chem Phys 38 (1963) 253, doi 10.1063/1.1733473.
"""
import math
import os

import numpy as np

R = 8.314462618
PA_PER_BAR = 1.0e5
HERE = os.path.dirname(os.path.abspath(__file__))
VAULT = os.path.join(HERE, "..", "data", "thermo")  # bundled thermochemical inputs

T_RS25, T_F1, T_EDGE = 3248.0, 3143.0, 2200.0


def _lnT(temps, vals, T):
    A = np.vstack([np.log(temps), np.ones(len(temps))]).T
    c, *_ = np.linalg.lstsq(A, np.array(vals), rcond=None)
    return c[0] * math.log(T) + c[1]


def _janaf_gef(code, T):
    """gef from a saved JANAF table, linearly interpolated on the tabulated grid."""
    rows = []
    for line in open(os.path.join(VAULT, "_gases", "raw", f"{code}.txt"), encoding="utf-8"):
        f = line.split("\t")
        try:
            rows.append((float(f[0]), float(f[3])))
        except (ValueError, IndexError):
            continue
    return float(np.interp(T, [r[0] for r in rows], [r[1] for r in rows]))


# HfO(g) Gibbs energy function gef = S - (H-H298)/T, J/mol/K, on the 1500-2500 K grid the
# fit uses. Transcribed from the supplementary of Bauschlicher, Kowalski and Jacobson,
# J Chem Phys 157 (2022) 154302, doi 10.1063/5.0120504 (the printed Cp equation is misprinted
# and is not used). Extrapolated in lnT above 2500 K.
_HFO_G_GEF = (
    [1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500],
    [263.0155, 264.8636, 266.6314, 268.3254, 269.9516, 271.5153, 273.0216, 274.4749,
     275.8796, 277.2392, 278.5572],
)


def _hfo_gef(T):
    return _lnT(_HFO_G_GEF[0], _HFO_G_GEF[1], T)


# Barin gef ladders, transcribed from page renders with the values verified in
# docs/JANAF_BARIN_extraction_2026-07-18.md
_BARIN = {
    "HfO2_c":  ([2200, 3173, 3200, 3300], [143.766, 173.866, 174.880, 178.556]),
    "ZrO2_c":  ([2200, 2950, 3200, 3300], [130.400, 150.183, 158.116, 161.101]),
    "ZrO2_g":  ([2200, 3200, 3300, 3500], [333.674, 352.327, 353.906, 356.943]),
    "ZrO_g":   ([2200, 2400, 2500],       [269.699, 273.106, 274.734]),
    "Ta2O5_c": ([2200, 3000],             [334.031, 401.954]),
    "TaO2_g":  ([2200, 2300, 3000],       [338.949, 341.080, 354.210]),
    "TaO_g":   ([2200, 3000],             [280.711, 291.437]),
}


def gef(key, T):
    ts, vs = _BARIN[key]
    if T <= ts[-1]:
        return float(np.interp(T, ts, vs))
    return _lnT(ts, vs, T)          # beyond the table, extrapolate in lnT


# dfH(298), kJ/mol
DFH = {
    "HfO2_c": -1117.63,   # Kornilov 1975, NOT Barin's superseded -1144.742
    "HfO_g":    63.19,    # Bauschlicher 2022
    "O_g":     249.173,   # JANAF O-001
    "ZrO2_c": -1097.463, "ZrO2_g": -286.186, "ZrO_g": 58.576,
    "Ta2O5_c": -2045.976, "TaO2_g": -200.832, "TaO_g": 192.464,
}


def _K(drH_kJ, dgef, T, dH=0.0):
    """dH is an ADDITIVE shift on the reaction enthalpy, kJ/mol, for uncertainty
    propagation. Additive rather than fractional because the sourced uncertainties are
    quoted that way (Kornilov +/- 1.6, Bauschlicher +/- 10) and because a fractional
    scaling of a 1400 kJ/mol reaction enthalpy is meaninglessly large.
    """
    return math.exp(-(drH_kJ + dH) * 1000.0 / (R * T) + dgef / R)


def p_hf(T, dH=0.0):
    """HfO(g) over hafnia, congruent so p(HfO) = p(O) balanced against O2."""
    K2 = _K(DFH["HfO_g"] + DFH["O_g"] - DFH["HfO2_c"],
            _hfo_gef(T) + _janaf_gef("O-001", T) - gef("HfO2_c", T), T, dH)
    K3 = (10.0 ** _janaf_logKf(T)) ** 2
    roots = np.roots([2.0 / K3, 1.0, 0.0, -K2])
    p_O = max(r.real for r in roots if abs(r.imag) < 1e-12 and r.real > 0)
    return (K2 / p_O) * PA_PER_BAR


def _janaf_logKf(T):
    rows = []
    for line in open(os.path.join(VAULT, "_gases", "raw", "O-001.txt"), encoding="utf-8"):
        f = line.split("\t")
        try:
            rows.append((float(f[0]), float(f[7])))
        except (ValueError, IndexError):
            continue
    return float(np.interp(T, [r[0] for r in rows], [r[1] for r in rows]))


def p_zr(T, dH=0.0):
    """Both zirconia channels, coupled through the shared oxygen balance."""
    K1 = _K(DFH["ZrO2_g"] - DFH["ZrO2_c"], gef("ZrO2_g", T) - gef("ZrO2_c", T), T, dH)
    K2 = _K(DFH["ZrO_g"] + DFH["O_g"] - DFH["ZrO2_c"],
            gef("ZrO_g", T) + _janaf_gef("O-001", T) - gef("ZrO2_c", T), T, dH)
    K3 = (10.0 ** _janaf_logKf(T)) ** 2
    roots = np.roots([2.0 / K3, 1.0, 0.0, -K2])
    p_O = max(r.real for r in roots if abs(r.imag) < 1e-12 and r.real > 0)
    return (K1 + K2 / p_O) * PA_PER_BAR


def p_hfzr(T, x_hf, dH_hf=0.0, dH_zr=0.0, detail=False):
    """Metal-bearing pressure over a mixed hafnia-zirconia scale, coupled through one
    shared oxygen potential rather than summed as independent pure oxides.

    Hafnia and zirconia sit in the same gas phase, so their dissociations share a single
    p(O). Zirconia is the more volatile, so its released oxygen raises p(O) and suppresses
    the hafnia monoxide channel, a common-oxygen effect a Raoult sum of the two pure-oxide
    pressures misses. The three oxygen-consuming monoxide channels enter one cubic in p(O);
    the stoichiometric ZrO2(g) channel carries its own oxygen and stays out of the balance.

    The oxide activities are the metal fractions on the (Hf,Zr) sublattice, a = x for a
    near-ideal solution (Section 3.6). At x_hf = 1 this returns p_hf(T) and at x_hf = 0 it
    returns p_zr(T) exactly, so the pure limits are the resolved values, unchanged.
    """
    a_hf, a_zr = x_hf, 1.0 - x_hf
    K2_hf = _K(DFH["HfO_g"] + DFH["O_g"] - DFH["HfO2_c"],
               _hfo_gef(T) + _janaf_gef("O-001", T) - gef("HfO2_c", T), T, dH_hf)
    K1_zr = _K(DFH["ZrO2_g"] - DFH["ZrO2_c"], gef("ZrO2_g", T) - gef("ZrO2_c", T), T, dH_zr)
    K2_zr = _K(DFH["ZrO_g"] + DFH["O_g"] - DFH["ZrO2_c"],
               gef("ZrO_g", T) + _janaf_gef("O-001", T) - gef("ZrO2_c", T), T, dH_zr)
    K3 = (10.0 ** _janaf_logKf(T)) ** 2
    # shared oxygen: the two monoxide channels feed one cubic 2/K3 pO^3 + pO^2 - K2sum = 0
    K2sum = K2_hf * a_hf + K2_zr * a_zr
    roots = np.roots([2.0 / K3, 1.0, 0.0, -K2sum])
    p_O = max(r.real for r in roots if abs(r.imag) < 1e-12 and r.real > 0)
    p_hfo = K2_hf * a_hf / p_O
    p_zro = K2_zr * a_zr / p_O
    p_zro2 = K1_zr * a_zr
    p_hf_bearing = p_hfo * PA_PER_BAR
    p_zr_bearing = (p_zro + p_zro2) * PA_PER_BAR
    if detail:
        return p_hf_bearing + p_zr_bearing, p_hf_bearing, p_zr_bearing
    return p_hf_bearing + p_zr_bearing


def p_ta(T, dH=0.0):
    """TaO2(g) over molten tantala, incongruent: Ta2O5(l) = 2 TaO2(g) + 1/2 O2(g).

    Stoichiometry fixes p(O2) = p(TaO2)/4, so K = p(TaO2)^2 (p(TaO2)/4)^0.5.
    TaO(g) is added afterwards; it runs about 70x below TaO2 at 2200 K.
    """
    drH = 2 * DFH["TaO2_g"] - DFH["Ta2O5_c"]
    dgef = 2 * gef("TaO2_g", T) + 0.5 * _janaf_gef("O-029", T) - gef("Ta2O5_c", T)
    K = _K(drH, dgef, T, dH)
    p_TaO2 = (K * 2.0) ** (1.0 / 2.5)
    drH_m = DFH["TaO_g"] + DFH["O_g"] - DFH["Ta2O5_c"] / 2.0
    dgef_m = gef("TaO_g", T) + _janaf_gef("O-001", T) - gef("Ta2O5_c", T) / 2.0
    p_TaO = math.sqrt(max(_K(drH_m, dgef_m, T, dH), 0.0))
    return (p_TaO2 + p_TaO) * PA_PER_BAR


def p_nb(T, oxidising=False, dH=0.0):
    """Unchanged from the original work; this branch always rested on primary data."""
    if oxidising:
        return 10.0 ** (-0.889e4 / T + 3.700)
    dgef = -0.012337 * T + 179.710
    return 1.4 * math.exp(-(546.5e3 + dH*1000.0) / (R * T) + dgef / R) * 1.0e5


P = {"Hf": p_hf, "Zr": p_zr, "Ta": p_ta,
     "Nb": lambda T, dH=0.0: p_nb(T, T < 2500, dH)}


def main():
    print("=" * 78)
    print("Volatile metal pressures on sourced thermochemistry")
    print("=" * 78)
    for T, lab in [(T_RS25, "RS-25 throat"), (T_F1, "F-1 throat"), (T_EDGE, "leading edge")]:
        print(f"\n  {lab}, {T:.0f} K")
        for m in ("Hf", "Zr", "Ta", "Nb"):
            print(f"    p({m}) = {P[m](T):12.4g} Pa")
    print()
    print("  cross-check, hafnia at the RS-25 throat")
    print(f"    this work (Kornilov + Bauschlicher) : {p_hf(T_RS25):.1f} Pa")
    print("    Panish 1963 measured pressures       : 40.3 Pa")
    print("    manuscript vendor anchor             : 1.06 Pa")


if __name__ == "__main__":
    main()
