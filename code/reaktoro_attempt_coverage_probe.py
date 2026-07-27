#!/usr/bin/env python3
# Documents the failed Reaktoro/NASA-CEA database attempt; Hf/Zr/Ta/Nb species absent. Results are in oxide_vaporisation_results.json (Raoult sum via vapor_equilibrium.py).
"""
Odinzen engine :: real multiphase equilibrium for the four Hf and Zr rich
survivors, replacing the step 2 scorecard with a true Gibbs energy minimisation.

Engine  : Reaktoro (conda only).   Data : NASA CEA (bundled with Reaktoro).
Targets : RS-25 throat, F-1 throat, hypersonic leading edge.

How to run (Ubuntu, conda miniforge)
------------------------------------
    conda activate odinzen-rkt
    python reaktoro_uhtc_equilibrium.py

Outputs a printed report and oxide_vaporisation_results.json next to this file.
"""
# This script documents the abandoned Gibbs-minimiser attempt. NASA-CEA carries no Hf/Zr/Ta/Nb
# species, so no equilibrium could be posed. The actual results (oxide vaporisation Raoult sum)
# are in oxide_vaporisation_results.json, produced by vapor_equilibrium.py.
from __future__ import annotations
import json, sys

try:
    from reaktoro import (NasaDatabase, GaseousPhase, CondensedPhases,
                          CondensedPhase, ChemicalSystem, ChemicalState,
                          EquilibriumSolver, speciate, AggregateState)
    import numpy as np
except Exception as exc:  # pragma: no cover
    sys.exit("Reaktoro import failed. Install with:\n"
             "  conda create -n odinzen-rkt -c conda-forge python=3.11 reaktoro numpy -y\n"
             f"  conda activate odinzen-rkt\nUnderlying error: {exc}")

CANDIDATES = {
    "C2_HfZr":   ["Hf", "Zr"],
    "C3_HfZrTa": ["Hf", "Zr", "Ta"],
    "C3_HfZrNb": ["Hf", "Zr", "Nb"],
    "C4_noTi":   ["Hf", "Zr", "Ta", "Nb"],
}
CERAMIC_MOL = 1.0e-3
GAS_MOL     = 1.0

ENVIRONMENTS = {
    "Target1a_RS25_throat": dict(
        T=3248.0, P_bar=116.3,
        gas={"H2O": 0.75, "H2": 0.25},
        label="RS-25 throat (LOX/LH2, O/F 6, wet reducing, M=1)"),
    "Target1b_F1_throat": dict(
        T=3143.0, P_bar=39.4,
        gas={"H2O": 0.32, "CO": 0.34, "CO2": 0.10, "H2": 0.24},
        label="F-1 throat (LOX/RP-1, O/F 2.27, carbon rich, M=1)"),
    "Target2_leadingedge": dict(
        T=2200.0, P_bar=0.05,
        gas={"O2": 0.21, "N2": 0.79},
        label="Hypersonic leading edge (air, dry)"),
}

METALS_ALL = ["Hf", "Zr", "Ta", "Nb"]


def _is_gas(sp):
    try:
        return sp.aggregateState() == AggregateState.Gas
    except Exception:
        return False


def build_system(elements):
    el = " ".join(sorted(set(elements + ["C", "O", "H", "N"])))
    db = NasaDatabase("nasa-cea")
    gas = GaseousPhase(speciate(el))
    condensed = CondensedPhases(speciate(el))
    system = ChemicalSystem(db, gas, condensed)
    return db, system


def _seed(state, system, symbol, moles, allow_substring):
    """Seed an element by setting the simplest species in the system that
    contains it. Match by species name (robust across Reaktoro versions):
    first an exact base name (e.g. 'Hf', 'Hf(cr)'); then, for metals, any name
    containing the symbol (e.g. 'HfO2(cr)'). Prefer condensed, then shortest."""
    exact = [s for s in system.species() if s.name().split("(")[0] == symbol]
    pool = exact
    if not pool and allow_substring:
        pool = [s for s in system.species() if symbol in s.name()]
    if not pool:
        raise RuntimeError(f"no species containing element {symbol} in system")
    pool = sorted(pool, key=lambda s: (_is_gas(s), len(s.name())))
    state.set(pool[0].name(), moles, "mol")
    return pool[0].name()


def run_case(name, metals, env):
    db, system = build_system(metals)
    state = ChemicalState(system)
    state.temperature(env["T"], "kelvin")
    state.pressure(env["P_bar"], "bar")

    per_metal = CERAMIC_MOL / len(metals)
    seeded = {}
    for m in metals:
        seeded[m] = _seed(state, system, m, per_metal, allow_substring=True)
    seeded["C"] = _seed(state, system, "C", CERAMIC_MOL, allow_substring=False)
    for sp, frac in env["gas"].items():
        state.set(sp, frac * GAS_MOL, "mol")

    solver = EquilibriumSolver(system)
    result = solver.solve(state)
    converged = bool(result.optima.succeeded) if hasattr(result, "optima") else True

    n = np.array(state.speciesAmounts())
    P_Pa = env["P_bar"] * 1.0e5
    gas_idx = [(i, sp.name()) for i, sp in enumerate(system.species()) if _is_gas(sp)]
    cond_idx = [(i, sp.name()) for i, sp in enumerate(system.species()) if not _is_gas(sp)]
    n_gas_tot = sum(n[i] for i, _ in gas_idx) or 1.0

    scale = {nm: float(n[i]) for i, nm in cond_idx if n[i] > 1e-12}
    partial = {}
    for i, nm in gas_idx:
        if n[i] <= 0:
            continue
        partial[nm] = (n[i] / n_gas_tot) * P_Pa

    def has_metal(s):
        return any(m in s for m in METALS_ALL)
    vol_metal = {k: v for k, v in partial.items() if has_metal(k)}
    p_metal_sum = sum(vol_metal.values())
    top_metal = dict(sorted(vol_metal.items(), key=lambda kv: kv[1], reverse=True)[:6])
    top_all = dict(sorted(partial.items(), key=lambda kv: kv[1], reverse=True)[:8])

    return dict(converged=converged, scale=scale, seeded=seeded,
                p_metal_volatile_Pa=p_metal_sum,
                top_metal_volatiles_Pa=top_metal,
                top_gas_species_Pa=top_all)


def main():
    payload = {"environments": {k: v["label"] for k, v in ENVIRONMENTS.items()},
               "results": {}}
    # coverage probe: list species the database carries for each metal
    db = NasaDatabase("nasa-cea")
    print("=" * 84)
    print("NASA CEA coverage probe (species whose name contains the metal symbol):")
    for m in METALS_ALL:
        names = [s.name() for s in db.species() if m in s.name()]
        print(f"  {m:3} : {', '.join(names[:10]) if names else 'NONE FOUND'}")
    print("=" * 84)

    for envkey, env in ENVIRONMENTS.items():
        print(f"\n### {env['label']}  |  T={env['T']:.0f} K, P={env['P_bar']} bar")
        rows = []
        for name, metals in CANDIDATES.items():
            try:
                r = run_case(name, metals, env)
            except Exception as exc:
                print(f"  {name:10} ERROR: {exc}")
                payload["results"].setdefault(envkey, {})[name] = {"error": str(exc)}
                continue
            scale = "+".join(sorted(r["scale"])) or "none"
            print(f"  {name:10} scale: {scale:30} "
                  f"sum p(metal vol)={r['p_metal_volatile_Pa']:.3e} Pa  conv={r['converged']}")
            rows.append((name, r["p_metal_volatile_Pa"]))
            payload["results"].setdefault(envkey, {})[name] = r
        if rows:
            rows.sort(key=lambda kv: kv[1])
            print("  RANK (least volatile first): " + " > ".join(k for k, _ in rows))

    with open("oxide_vaporisation_results.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print("\n[written] oxide_vaporisation_results.json")


if __name__ == "__main__":
    main()
