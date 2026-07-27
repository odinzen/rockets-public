#!/usr/bin/env python3
"""
Real throat transport for two reference nozzles, RS-25 and F-1.

No placeholders. Each number traces to published engine parameters plus
standard transport theory:
  - throat state from the isentropic M = 1 relations,
  - binary diffusivity D and gas viscosity mu from Chapman Enskog kinetic
    theory with Neufeld collision integrals,
  - Sherwood number from the Bartz type turbulent pipe correlation with the
    Chilton Colburn heat to mass analogy:  Sh = 0.026 Re^0.8 Sc^(1/3),
  - length scale L = throat diameter,
  - mass transfer coefficient h_m = Sh * D / L.

Engine parameters (published)
  RS-25 : Pc 206 bar, Tc 3573 K, throat dia 0.262 m, O/F 6 (fuel rich),
          gas mostly H2O + excess H2, mean M about 13.6 g/mol, gamma 1.20.
  F-1   : Pc 70 bar,  Tc 3473 K, throat dia 0.88 m, O/F 2.27,
          gas H2O, CO, CO2, H2, mean M about 22.5 g/mol, gamma 1.21.
Sources: RS-25 Wikipedia / L3Harris spec sheet; F-1 Wikipedia / astronautix.
"""
from __future__ import annotations
import math

R = 8.314462618
NA = 6.02214076e23

def omega_D(Tstar):
    return (1.06036 / Tstar**0.15610 + 0.19300 / math.exp(0.47635 * Tstar)
            + 1.03587 / math.exp(1.52996 * Tstar) + 1.76474 / math.exp(3.89411 * Tstar))

def omega_mu(Tstar):
    return (1.16145 / Tstar**0.14874 + 0.52487 / math.exp(0.77320 * Tstar)
            + 2.16178 / math.exp(2.43787 * Tstar))

def viscosity(M_g, T, sigma, eps_k):
    """Chapman Enskog single component viscosity, Pa s. M_g g/mol, sigma Angstrom."""
    Om = omega_mu(T / eps_k)
    return 2.6693e-6 * math.sqrt(M_g * T) / (sigma**2 * Om)

def diffusivity(MA, MB, T, P_atm, sigA, sigB, epsA, epsB):
    """Chapman Enskog binary diffusivity, m^2/s. P in atm, sigma Angstrom, M g/mol."""
    sigAB = 0.5 * (sigA + sigB)
    epsAB = math.sqrt(epsA * epsB)
    Om = omega_D(T / epsAB)
    D_cm2 = (0.0018583 * math.sqrt(T**3 * (1.0 / MA + 1.0 / MB))
             / (P_atm * sigAB**2 * Om))
    return D_cm2 * 1.0e-4

# Lennard Jones parameters (literature typical; documented assumption)
LJ = {
    "volatile_oxide": dict(M=150.0, sigma=3.8, eps_k=900.0),   # ZrO/HfO heavy oxide
    "RS25_gas":       dict(M=13.6,  sigma=2.8, eps_k=300.0),   # H2O + H2 mix
    "F1_gas":         dict(M=22.5,  sigma=3.6, eps_k=320.0),   # H2O/CO/CO2/H2 mix
}

ENGINES = {
    "RS-25": dict(Pc_bar=206.0, Tc=3573.0, Dthroat=0.262, gamma=1.20,
                  gas="RS25_gas", note="LOX/LH2, O/F 6, wet reducing throat"),
    "F-1":   dict(Pc_bar=70.0,  Tc=3473.0, Dthroat=0.88,  gamma=1.21,
                  gas="F1_gas",  note="LOX/RP-1, O/F 2.27, carbon rich throat"),
}

# recession constants (shared with recession_model.py)
RHO_CERAMIC = 9.5e3
M_CERAMIC   = 0.147
VM_PER_METAL = M_CERAMIC / RHO_CERAMIC
UM_PER_HR = 1e6 * 3600.0


def throat_state(eng):
    g = eng["gamma"]
    Tt = eng["Tc"] / (1.0 + 0.5 * (g - 1.0))
    Pt_bar = eng["Pc_bar"] * (2.0 / (g + 1.0)) ** (g / (g - 1.0))
    gas = LJ[eng["gas"]]
    M = gas["M"] / 1000.0                       # kg/mol
    Rspec = R / M
    rho = Pt_bar * 1e5 * M / (R * Tt)           # kg/m3
    a = math.sqrt(g * Rspec * Tt)               # sonic velocity, throat
    mu = viscosity(gas["M"], Tt, gas["sigma"], gas["eps_k"])
    ox = LJ["volatile_oxide"]
    D = diffusivity(ox["M"], gas["M"], Tt, Pt_bar * 1e5 / 101325.0,
                    ox["sigma"], gas["sigma"], ox["eps_k"], gas["eps_k"])
    Re = rho * a * eng["Dthroat"] / mu
    Sc = mu / (rho * D)
    Sh = 0.026 * Re**0.8 * Sc**(1.0 / 3.0)
    h_m = Sh * D / eng["Dthroat"]
    # recession per unit volatile pressure, um/h per Pa (alpha independent)
    v_per_Pa = (h_m / (R * Tt)) * VM_PER_METAL * UM_PER_HR
    return dict(Tt=Tt, Pt_bar=Pt_bar, rho=rho, a=a, mu=mu, D=D, Re=Re,
                Sc=Sc, Sh=Sh, h_m=h_m, v_per_Pa=v_per_Pa)


def main():
    print("=" * 86)
    print("REAL THROAT TRANSPORT  (Chapman Enskog + Bartz/Chilton Colburn, M=1 throat)")
    print("=" * 86)
    rows = {}
    for name, eng in ENGINES.items():
        s = throat_state(eng)
        rows[name] = s
        print(f"\n{name}  ({eng['note']})")
        print(f"  throat T              : {s['Tt']:.0f} K")
        print(f"  throat P              : {s['Pt_bar']:.1f} bar")
        print(f"  gas density           : {s['rho']:.2f} kg/m3")
        print(f"  sonic velocity        : {s['a']:.0f} m/s")
        print(f"  gas viscosity         : {s['mu']:.2e} Pa s")
        print(f"  binary diffusivity D  : {s['D']:.2e} m2/s  (volatile oxide in gas)")
        print(f"  Reynolds (throat dia) : {s['Re']:.2e}")
        print(f"  Schmidt               : {s['Sc']:.3f}")
        print(f"  Sherwood              : {s['Sh']:.3e}")
        print(f"  mass transfer h_m     : {s['h_m']:.3f} m/s")
        print(f"  recession coefficient : {s['v_per_Pa']:.3f} um/h per Pa of volatile metal")
    print("\n" + "=" * 86)
    print("Use: throat recession um/h = (coefficient) x (equilibrium volatile metal pressure, Pa)")
    print("=" * 86)
    return rows


if __name__ == "__main__":
    main()
