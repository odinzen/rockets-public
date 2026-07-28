#!/usr/bin/env python3
"""
Configurational-entropy stability screen for the HEC-UHTC manuscript (Section 2.1 / SI S7).

Pair mixing enthalpies are the ten (Hf,Zr,Ti,Ta,Nb)C binaries computed in this work by
DFT (Quantum ESPRESSO 7.3.1, PBE, ONCV SG15 pseudopotentials; vc-relax then single point).
Pure endpoints use a 2-atom primitive FCC cell (1 f.u.); mixed pairs a 8-atom conventional
cubic cell (4 f.u.) with 50/50 metal ordering on the FCC sublattice. dHmix(0.5) is
E(mixed)/4 - 0.5 E(AC) - 0.5 E(BC). These replace the earlier back-calibrated set.

MODEL (manuscript Section 2.1):
    g(x) = 0.5 * x^T L x + R T * sum_i x_i ln x_i        (per formula unit)
    L_ij = 4 * dHmix_ij(0.5)
    (at 0 K g_mix = H_mix, which fixes L without new simulation)

SPINODAL TEST: single-phase stability is read from the eigenvalues of the Gibbs Hessian on
the composition simplex; the temperature at which the smallest eigenvalue crosses zero is
the equiatomic spinodal. A Hessian that is positive definite at all T >= 0 (every pair
stabilising) means no miscibility gap -- the solution is single-phase at every temperature.

The Hessian of g on the (n-1)-dim simplex at the equiatomic point x_i = 1/n. Eliminate the
last component (x_n = 1 - sum_{i<n} x_i) and build the reduced (n-1)x(n-1) Hessian H w.r.t.
the free fractions y = (x_1..x_{n-1}):

  Entropy part:  g_entropy = R T sum x_i ln x_i
    d2(g_ent)/dy_a dy_b = R T * ( delta_ab / x_a + 1 / x_n )
  Enthalpy part: U = 0.5 x^T L x, with x_n = 1 - sum y
    d2U/dy_a dy_b = L_ab - L_a,n - L_n,b + L_n,n
  At equiatomic x_i = 1/n.

Eigenvalues of the small symmetric matrix are found by a pure-Python Jacobi rotation
routine -- NO numpy.linalg (np.eig/lstsq/polyfit HARD-CRASH on this box).
"""
import math

R = 8.314462618
METALS5 = ["Hf", "Zr", "Ti", "Ta", "Nb"]

# dHmix(0.5) in J/mol per f.u., DFT this work (QE 7.3.1, PBE, ONCV SG15). Positive = demixing.
# These are the Sol-computed pair enthalpies; the openly-deposited copy with full method
# provenance is data/dft_pair_enthalpies.csv (kept in step with the values below).
DFT_DHMIX = {
    frozenset(("Hf", "Zr")):  -0.4e3,
    frozenset(("Hf", "Ti")): +12.0e3,
    frozenset(("Hf", "Ta")):  -4.8e3,
    frozenset(("Hf", "Nb")):  -1.5e3,
    frozenset(("Zr", "Ti")): +17.3e3,
    frozenset(("Zr", "Ta")):  -2.9e3,
    frozenset(("Zr", "Nb")):  +0.1e3,
    frozenset(("Ti", "Ta")): -13.9e3,
    frozenset(("Ti", "Nb")):  -7.1e3,
    frozenset(("Ta", "Nb")):  +2.6e3,
}

# ---------- pure-Python symmetric eigenvalues (Jacobi) ----------
def jacobi_eigenvalues(A, iters=100, tol=1e-14):
    n = len(A)
    a = [row[:] for row in A]
    for _ in range(iters):
        # largest off-diagonal
        p, q, mx = 0, 1, 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(a[i][j]) > mx:
                    mx = abs(a[i][j]); p, q = i, j
        if mx < tol:
            break
        app, aqq, apq = a[p][p], a[q][q], a[p][q]
        if abs(apq) < 1e-300:
            break
        phi = 0.5 * math.atan2(2 * apq, aqq - app)
        c, s = math.cos(phi), math.sin(phi)
        for k in range(n):
            akp, akq = a[k][p], a[k][q]
            a[k][p] = c * akp - s * akq
            a[k][q] = s * akp + c * akq
        for k in range(n):
            akp, akq = a[p][k], a[q][k]
            a[p][k] = c * akp - s * akq
            a[q][k] = s * akp + c * akq
    return [a[i][i] for i in range(n)]

# ---------- reduced simplex Hessian at equiatomic ----------
def reduced_hessian(L, T, n):
    """H_ab = d2 g / dy_a dy_b, a,b in 0..n-2, at x_i = 1/n."""
    x = 1.0 / n
    Ln = n - 1  # index of eliminated component
    H = [[0.0] * (n - 1) for _ in range(n - 1)]
    for a in range(n - 1):
        for b in range(n - 1):
            # enthalpy: L_ab - L_a,n - L_n,b + L_n,n
            enth = L[a][b] - L[a][Ln] - L[Ln][b] + L[Ln][Ln]
            # entropy: R T (delta_ab / x_a + 1 / x_n); equiatomic -> R T (n*delta + n)
            ent = R * T * (n * (1.0 if a == b else 0.0) + n)
            H[a][b] = enth + ent
    return H

def min_eigenvalue(L, T, n):
    H = reduced_hessian(L, T, n)
    return min(jacobi_eigenvalues(H))

def spinodal(L, n, Tlo=1.0, Thi=20000.0):
    """Lowest T at which min eigenvalue >= 0 (stable above). Bisection on the
    sign of the smallest eigenvalue; returns the crossing T, or <Tlo if already
    stable at Tlo (i.e. no instability -> report '< Tlo')."""
    flo = min_eigenvalue(L, n=n, T=Tlo)
    if flo >= 0:
        return Tlo, True   # stable even at ~0 K floor -> effectively no spinodal
    fhi = min_eigenvalue(L, n=n, T=Thi)
    if fhi < 0:
        return Thi, False  # still unstable at ceiling
    for _ in range(200):
        Tm = 0.5 * (Tlo + Thi)
        fm = min_eigenvalue(L, n=n, T=Tm)
        if fm < 0:
            Tlo = Tm
        else:
            Thi = Tm
    return 0.5 * (Tlo + Thi), True

# ---------- build L from a pair-enthalpy dict ----------
def build_L(metals, dHmix):
    n = len(metals)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            key = frozenset((metals[i], metals[j]))
            L[i][j] = 4.0 * dHmix[key]
    return L

def run_set(label, dHmix):
    print(f"\n===== ENTHALPY SET: {label} =====")
    sets = {
        "C2_HfZr":   ["Hf", "Zr"],
        "C3_HfZrTa": ["Hf", "Zr", "Ta"],
        "C3_HfZrNb": ["Hf", "Zr", "Nb"],
        "C4_noTi":   ["Hf", "Zr", "Ta", "Nb"],
        "C5":        ["Hf", "Zr", "Ti", "Ta", "Nb"],
    }
    for name, metals in sets.items():
        n = len(metals)
        L = build_L(metals, dHmix)
        Ts, ok = spinodal(L, n)
        tag = f"{Ts:8.1f} K" if Ts > 1.5 else "   < 1 K (no unmixing)"
        print(f"  {name:10} (n={n})  spinodal = {tag}")

def report_spinodal(name, metals):
    n = len(metals)
    L = build_L(metals, DFT_DHMIX)
    Ts, ok = spinodal(L, n)
    if Ts <= 1.5:
        return name, None  # positive-definite at ~0 K: no miscibility gap
    return name, Ts


if __name__ == "__main__":
    # solver validation: the n=2 full-Hessian path must reproduce the binary
    # closed form T_spin = 2*dHmix(0.5)/R for a positive (demixing) pair.
    dH_pos = 10.0e3
    Lchk = [[0.0, 4 * dH_pos], [4 * dH_pos, 0.0]]
    Ts_chk, _ = spinodal(Lchk, 2)
    print(f"solver check: binary dHmix=+10 kJ/mol -> spinodal {Ts_chk:.1f} K "
          f"(closed form {2*dH_pos/R:.1f} K)")

    candidates = {
        "C2_HfZr":   ["Hf", "Zr"],
        "C3_HfZrTa": ["Hf", "Zr", "Ta"],
        "C3_HfZrNb": ["Hf", "Zr", "Nb"],
        "C4_noTi":   ["Hf", "Zr", "Ta", "Nb"],
        "C5":        ["Hf", "Zr", "Ti", "Ta", "Nb"],
    }
    old = {"C2_HfZr": 118, "C3_HfZrTa": 80, "C3_HfZrNb": 84, "C4_noTi": 68, "C5": 2138}

    print("\nDFT pair enthalpies (kJ/mol, this work):")
    for k, v in DFT_DHMIX.items():
        print(f"  {'-'.join(sorted(k)):8} {v/1e3:+6.1f}")

    print("\ncarbide candidate spinodals (DFT L_ij = 4 dHmix):")
    print(f"  {'candidate':11} {'DFT spinodal':>16}   {'old (back-calib)':>16}")
    for name, metals in candidates.items():
        _, Ts = report_spinodal(name, metals)
        dft = f"{Ts:8.1f} K" if Ts else "  none (stable all T)"
        print(f"  {name:11} {dft:>16}   {old[name]:>13} K")
