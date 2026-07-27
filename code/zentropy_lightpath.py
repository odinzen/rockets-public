#!/usr/bin/env python3
"""
Light-path zentropy refinement of the HEC-UHTC configurational entropy, run through
pyzentropy (Hew, Myers, Shang, Liu, arXiv:2604.17665). No new DFT: the configuration
energies come from the ten (Hf,Zr,Ti,Ta,Nb)C pair enthalpies already computed for the
stability screen (stability_screen.py).

Model. Metals sit on the FCC metal sublattice of the rocksalt carbide (12 nearest
neighbours). A 2x2x2 conventional supercell (32 sites, 192 NN bonds, correct coordination)
is decorated at the target composition; a decoration's energy is the sum over unlike NN
bonds of u_ij = L_ij / z, with L_ij = 4 dHmix_ij(0.5) and z = 12, which reproduces the
paper's mean-field regular solution in the random limit. Uniform random decorations sample
the configurational density of states g(E); each populated energy level becomes a pyzentropy
Configuration (flat in V and T, zero internal entropy) with multiplicity scaled to the true
multinomial count. pyzentropy's recursion then returns the configurational Helmholtz energy
and entropy of the solid solution. At high T this recovers the ideal R sum x ln x; below that
the short-range order implied by the pair enthalpies lowers F and trims S. The service-window
question is whether that trim ever threatens the single-phase state (it does not).
"""
import itertools
import math
import numpy as np

from pyzentropy.configuration import Configuration
from pyzentropy.system import System

R = 8.314462618
Z = 12  # FCC nearest neighbours
RNG = np.random.default_rng(20260717)

DFT_DHMIX = {
    frozenset(("Hf", "Zr")):  -0.4e3, frozenset(("Hf", "Ti")): +12.0e3,
    frozenset(("Hf", "Ta")):  -4.8e3, frozenset(("Hf", "Nb")):  -1.5e3,
    frozenset(("Zr", "Ti")): +17.3e3, frozenset(("Zr", "Ta")):  -2.9e3,
    frozenset(("Zr", "Nb")):  +0.1e3, frozenset(("Ti", "Ta")): -13.9e3,
    frozenset(("Ti", "Nb")):  -7.1e3, frozenset(("Ta", "Nb")):  +2.6e3,
}


def u_matrix(metals):
    n = len(metals)
    U = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                U[i, j] = 4.0 * DFT_DHMIX[frozenset((metals[i], metals[j]))] / Z
    return U


def fcc_conventional(reps=2):
    """reps^3 conventional FCC cells -> 4*reps^3 sites, periodic NN bond list."""
    basis = [(0, 0, 0), (.5, .5, 0), (.5, 0, .5), (0, .5, .5)]
    sites = [np.array([bi[0] + i, bi[1] + j, bi[2] + k])
             for i in range(reps) for j in range(reps) for k in range(reps) for bi in basis]
    sites = np.array(sites)
    n = len(sites)
    edge = float(reps)
    nn = 1e9
    dif = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                dif[i, j] = 1e9
                continue
            d = sites[i] - sites[j]
            d -= edge * np.round(d / edge)
            dd = math.sqrt(d @ d)
            dif[i, j] = dd
            nn = min(nn, dd)
    bonds = [(i, j) for i in range(n) for j in range(i + 1, n)
             if abs(dif[i, j] - nn) < 1e-6]
    return n, np.array(bonds)


def density_of_states(metals, counts, bonds, n_sites, nsamp=1200000, nbins=64):
    """Sample uniform random decorations; return energy-level centres (J/mol/site) and
    multiplicities scaled to the true multinomial count W."""
    U = u_matrix(metals)
    base = np.concatenate([np.full(c, k) for k, c in enumerate(counts)])
    bi, bj = bonds[:, 0], bonds[:, 1]
    energies = np.empty(nsamp)
    for s in range(nsamp):
        perm = RNG.permutation(base)
        energies[s] = U[perm[bi], perm[bj]].sum() / n_sites
    hist, edges = np.histogram(energies, bins=nbins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    mask = hist > 0
    centres, hist = centres[mask], hist[mask]
    # true total count W (multinomial), scale sampled histogram to it
    logW = math.lgamma(n_sites + 1) - sum(math.lgamma(c + 1) for c in counts)
    W = math.exp(logW)
    mult = (hist / nsamp) * W
    return centres, mult, W


EV = 96485.33212  # J/mol per eV; pyzentropy works in eV/atom with k_B in eV/K


def zentropy_S(E, mult, T, n_sites):
    """E in J/mol/site. Feed pyzentropy whole-cell energies (eV/cell) with the cell
    multiplicities, then divide results by n_sites to report per mole of metal sites."""
    E_ev = (E * n_sites) / EV  # per-cell energy in eV
    volumes = np.array([0.0, 1.0, 2.0])
    nV, nT = len(volumes), len(T)
    configs = {}
    for k, (Ek, gk) in enumerate(zip(E_ev, mult)):
        F = np.full((nT, nV), Ek)
        Zz = np.zeros((nT, nV))
        configs[f"c{k}"] = Configuration(
            name=f"c{k}", multiplicity=max(int(round(gk)), 1), number_of_atoms=n_sites,
            volumes=volumes, temperatures=T.astype(float),
            helmholtz_energies=F, helmholtz_energies_dV=Zz, helmholtz_energies_d2V2=Zz,
            entropies=Zz, heat_capacities=Zz,
        )
    gs = f"c{int(np.argmin(E_ev))}"
    sysm = System(configurations=configs, ground_state=gs)
    sysm.calculate_partition_functions()
    sysm.calculate_probabilities()
    sysm.calculate_helmholtz_energies(configs[gs].helmholtz_energies)
    sysm.calculate_entropies()
    vi = nV // 2
    F_sys = np.asarray(sysm.helmholtz_energies)[:, vi] * EV / n_sites   # -> J/mol site
    S_sys = np.asarray(sysm.entropies)[:, vi] * EV / n_sites            # -> J/mol/K site
    return F_sys, S_sys


CANDIDATES = {
    "C2_HfZr":   (["Hf", "Zr"],             [16, 16]),
    "C3_HfZrTa": (["Hf", "Zr", "Ta"],       [11, 11, 10]),
    "C3_HfZrNb": (["Hf", "Zr", "Nb"],       [11, 11, 10]),
    "C4_noTi":   (["Hf", "Zr", "Ta", "Nb"], [8, 8, 8, 8]),
    "C5":        (["Hf", "Zr", "Ti", "Ta", "Nb"], [7, 7, 6, 6, 6]),
}

LABELS = {
    "C2_HfZr": "(Hf,Zr)C", "C3_HfZrTa": "(Hf,Zr,Ta)C", "C3_HfZrNb": "(Hf,Zr,Nb)C",
    "C4_noTi": "(Hf,Zr,Ta,Nb)C", "C5": "(Hf,Zr,Ti,Ta,Nb)C",
}

if __name__ == "__main__":
    import json
    n_sites, bonds = fcc_conventional(2)
    print(f"FCC supercell: {n_sites} metal sites, {len(bonds)} NN bonds "
          f"(coordination {2*len(bonds)/n_sites:.0f})\n")
    T = np.linspace(1000.0, 3873.0, 40)   # service window up to the 3600 C benchmark
    out = {"temperatures_K": T.tolist(), "n_sites": n_sites, "candidates": {}}
    for name, (metals, counts) in CANDIDATES.items():
        x = np.array(counts) / sum(counts)
        S_bulk = -R * np.sum(x * np.log(x))
        E, mult, W = density_of_states(metals, counts, bonds, n_sites)
        S_cell_ideal = R * math.log(W) / n_sites  # finite-cell ideal (k lnW / N)
        F_sys, S_sys = zentropy_S(E, mult, T, n_sites)
        trim = S_cell_ideal - S_sys
        out["candidates"][name] = {
            "label": LABELS[name], "metals": metals, "counts": counts,
            "S_bulk_ideal": S_bulk, "S_cell_ideal": S_cell_ideal,
            "S_zentropy": S_sys.tolist(), "SRO_trim": trim.tolist(),
        }
        print(f"== {name}: {LABELS[name]} ==")
        print(f"   bulk-ideal S={S_bulk:6.3f}  finite-cell-ideal S={S_cell_ideal:6.3f} J/mol/K")
        for Ti in (1200.0, 2000.0, 3000.0, 3873.0):
            j = int(np.argmin(abs(T - Ti)))
            print(f"   T={T[j]:6.0f} K  zentropy S={S_sys[j]:6.3f}  "
                  f"SRO trim {trim[j]:+.4f} J/mol/K")
        print()
    dst = "code/zentropy_sro_results.json"
    json.dump(out, open(dst, "w"), indent=1)
    print("wrote", dst)
