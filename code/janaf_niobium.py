#!/usr/bin/env python3
"""
Third-law NbO2(g) pressure over liquid NbO2 at the throat temperatures, from the
JANAF free-energy functions, replacing the Arrhenius extrapolation.

Method (free energy function / third law):
  NbO2(l) = NbO2(g)
  dG(T) = dH_vap,298 - T * d_gef(T),   gef = -(G_T - H_298)/T
  ln(p/p0) = -dG(T)/(R T) = -dH_vap,298/(R T) + d_gef(T)/R      (p0 = 1 bar)

Data, all from Kamegashira, Matsui, Harada, Naito, J. Nucl. Mater. 101 (1981) 207:
  dH_vap,298 (NbO2 l -> g), third law      = 546.5 kJ/mol      (their Table 5)
  d_gef = gef(NbO2,g) - gef(NbO2,l), revised (their Table 8, liquid rows):
     2100 K : 153.81    2200 K : 152.55    2300 K : 151.31    2400 K : 150.11  (J/mol K)
The free energy function difference is fit linearly and extended to the throat;
this extends a smooth ~150 J/mol K quantity by ~850 K, not the exponential pressure.
"""
import numpy as np

R = 8.314462618
dH_vap_298 = 546.5e3            # J/mol, Kamegashira Table 5, third law

# d_gef (revised, liquid) vs T, Kamegashira Table 8
Tg = np.array([2100.0, 2200.0, 2300.0, 2400.0])
dgef = np.array([153.81, 152.55, 151.31, 150.11])      # J/mol K
def _linfit(xs, ys):
    n = len(xs); sx = sum(xs); sy = sum(ys)
    sxx = sum(x*x for x in xs); sxy = sum(xs[i]*ys[i] for i in range(n))
    slope = (n*sxy - sx*sy) / (n*sxx - sx*sx)
    return slope, (sy - slope*sx) / n
slope, intercept = _linfit(list(Tg), list(dgef))
def d_gef(T):
    return slope * T + intercept

def p_NbO2_thirdlaw(T):
    lnp = -dH_vap_298 / (R * T) + d_gef(T) / R          # p in bar
    return np.exp(lnp) * 1.0e5                            # Pa

def p_NbO2_arrhenius(T):
    # Kamegashira Table 4 liquid fit, extrapolated
    return 10.0 ** (-2.393e4 / T + 10.90)                # Pa

print("d_gef(T) linear fit:  d_gef = %.5f*T + %.3f  J/mol K" % (slope, intercept))
print("="*78)
print(f"{'T, K':>8} {'d_gef':>9} {'p third-law, Pa':>18} {'p Arrhenius, Pa':>18} {'ratio':>8}")
for T in [2400.0, 3143.0, 3248.0]:
    p3 = p_NbO2_thirdlaw(T); pa = p_NbO2_arrhenius(T)
    print(f"{T:8.0f} {d_gef(T):9.2f} {p3:18.1f} {pa:18.1f} {p3/pa:8.3f}")
print("="*78)
# metal-bearing total = NbO2(g) + NbO(g); measured NbO/NbO2 ratio ~ 0.4 -> x1.4
for T, eng in [(3248.0,"RS-25"), (3143.0,"F-1")]:
    p_metal_Nb = 1.4 * p_NbO2_thirdlaw(T)
    print(f"{eng:6} throat T={T:.0f} K :  p(NbO2,g) third-law = {p_NbO2_thirdlaw(T):.0f} Pa, "
          f"total Nb metal (x1.4 for NbO) = {p_metal_Nb:.0f} Pa")
