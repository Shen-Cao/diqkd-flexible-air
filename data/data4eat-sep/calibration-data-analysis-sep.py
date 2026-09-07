#!/usr/bin/env python3
r"""
**Calibration** data analysis for the separate-IR DIQKD pipeline (Sep).

Estimates the **assumed QBER** of the two separate-IR codecs from the lab
calibration count matrices only:

  (A) Calibration analysis
      using the updated CHSH counts (5997 rounds in the 4 standard Bell-test
      subgroups; 6086 counting all subgroups, S = 2.6130).
      ZZ counts and Gamma are unchanged.

  (B) Assumed-QBER estimates for the two codecs:
        - CHSH codec : upper bounds on QBER_E at 90/95/99/99.9%
        - ZZ codec   : upper bounds on Q at 90/95/99/99.9%
      All upper bounds are ceiled to 5e-4.

All inputs are the calibration count matrices hardcoded below; the program
reads no files.

Usage:
  python calibration-data-analysis-sep.py
  python calibration-data-analysis-sep.py --gamma 0.02
"""
import argparse
import math

import numpy as np

CONF_LEVELS = [(0.90, 1.644854), (0.95, 1.959964),
               (0.99, 2.575829), (0.999, 3.290527)]

# ============================================================
# Calibration data (updated — corrected laser-unlock artifact)
# Each row: [n00, n01, n10, n11] = joint outcome counts
# 4 standard CHSH subgroups (alice_basis, bob_basis) in {(0,0),(0,1),(1,0),(1,1)}
# ============================================================
CHSH_COUNTS = np.array([
    [156, 733, 798, 162],
    [766, 187, 144, 836],
    [471,  92,  87, 455],
    [440, 118,  91, 461],
])

# Keygen calibration: Z⊗Z measurement (unchanged)
ZZ_COUNTS = np.array([240, 5155, 5765, 134])


def _ceil_to_0005(x):
    """Round up to the nearest 0.0005 (5e-4, used for .pc BER naming)."""
    return math.ceil(x * 2000) / 2000


def _upper_bounds(point, sigma):
    """Yield (cl, upper) tuples, where upper = point + z_(1-a/2) * sigma."""
    for cl, z in CONF_LEVELS:
        yield cl, point + z * sigma


def _print_upper_ci(point, sigma):
    """Print a compact table of upper bounds and their 5e-4 ceilings."""
    print(f"      {'CL':>6}  {'upper':>16}  {'ceil(5e-4)':>12}")
    for cl, hi in _upper_bounds(point, sigma):
        print(f"      {cl*100:>5.1f}%  {hi*100:>15.4f}%  {_ceil_to_0005(hi):>12.4f}")


def analyze_calibrations(gamma):
    """Compute S, QBER_E, Q and their sigma from the calibration count matrices."""
    Es, ns_chsh = [], []
    total_qber_E_errors = 0
    for cells in CHSH_COUNTS:
        n = int(cells.sum())
        n00, n01, n10, n11 = cells
        agree = n00 + n11
        disagree = n01 + n10
        E = (agree - disagree) / n
        Es.append(E)
        ns_chsh.append(n)
        total_qber_E_errors += disagree if E > 0 else agree

    n_E = sum(ns_chsh)
    S = float(sum(abs(e) for e in Es))
    var_S = sum((1.0 - e * e) / n for e, n in zip(Es, ns_chsh) if n > 0)
    sigma_S = math.sqrt(var_S)
    QBER_E = total_qber_E_errors / n_E
    sigma_QBER_E = math.sqrt(QBER_E * (1.0 - QBER_E) / n_E)

    n_Q = int(ZZ_COUNTS.sum())
    n00, n01, n10, n11 = ZZ_COUNTS
    agree_kg = n00 + n11
    Q = float(agree_kg / n_Q)
    sigma_Q = math.sqrt(Q * (1.0 - Q) / n_Q)
    E_kg = float((agree_kg - (n_Q - agree_kg)) / n_Q)

    BER_combined = gamma * QBER_E + (1.0 - gamma) * Q
    sigma_tri = gamma * sigma_QBER_E + (1.0 - gamma) * sigma_Q
    sigma_indep = math.sqrt(gamma ** 2 * sigma_QBER_E ** 2 + (1 - gamma) ** 2 * sigma_Q ** 2)

    return {
        "n_E": n_E, "S": S, "sigma_S": sigma_S,
        "QBER_E": QBER_E, "sigma_QBER_E": sigma_QBER_E,
        "n_Q": n_Q, "Q": Q, "sigma_Q": sigma_Q, "E_kg": E_kg,
        "BER_combined": BER_combined,
        "sigma_tri": sigma_tri, "sigma_indep": sigma_indep,
    }


def print_calibration_section(r, gamma):
    print("\n[1] CHSH Calibration — 4 Bell-test subgroups")
    labels = ["(a=Z, b=Z+X)", "(a=Z, b=Z-X)", "(a=X, b=Z+X)", "(a=X, b=Z-X)"]
    print(f"    {'Subgroup':<18}{'cells [n00,n01,n10,n11]':<30}{'n':>6}{'E':>10}{'Flip?':>8}")
    print("    " + "-" * 78)
    for cells, lbl in zip(CHSH_COUNTS, labels):
        n = int(cells.sum())
        n00, n01, n10, n11 = cells
        E = (n00 + n11 - n01 - n10) / n
        flip = "YES" if E > 0 else "NO"
        print(f"    {lbl:<18}[{n00:>4},{n01:>4},{n10:>4},{n11:>4}]        "
              f"{n:>6}{E:>+10.4f}{flip:>8}")

    print(f"\n    S = {r['S']:.4f} +/- {r['sigma_S']:.4f}  (n_E = {r['n_E']})")
    print(f"    QBER_E (CHSH BER, optimal flip) = {r['QBER_E']:.6f} ({r['QBER_E']*100:.4f}%)")
    print(f"    sigma_QBER_E                    = {r['sigma_QBER_E']:.6f}")

    print(f"\n[2] Keygen Calibration — Z⊗Z measurement")
    n00, n01, n10, n11 = ZZ_COUNTS
    print(f"    cells = [{n00:>4},{n01:>4},{n10:>4},{n11:>4}]")
    print(f"    n_Q = {r['n_Q']}")
    print(f"    Q (BER_ZZ) = {r['Q']:.6f} ({r['Q']*100:.4f}%)")
    print(f"    sigma_Q    = {r['sigma_Q']:.6f}")
    print(f"    E_ZZ       = {r['E_kg']:+.4f}")

    print(f"\n[3] Legacy BER_combined (single-codec IR, kept for comparison)")
    print(f"    Gamma         = {gamma:.10f}")
    print(f"    BER_combined  = Gamma*QBER_E + (1-Gamma)*Q")
    print(f"                  = {r['BER_combined']:.6f} ({r['BER_combined']*100:.4f}%)")
    print(f"    sigma (triangle, no-independence) = {r['sigma_tri']:.6f}"
          f"  ({r['sigma_tri']*1e4:.2f} bps)")
    print(f"    sigma (independence)              = {r['sigma_indep']:.6f}"
          f"  ({r['sigma_indep']*1e4:.2f} bps)")
    print(f"    ratio (tri/indep)                 = {r['sigma_tri']/r['sigma_indep']:.4f}"
          f"  ({(r['sigma_tri']/r['sigma_indep']-1)*100:+.1f}%)")

    print(f"\n    BER_combined upper bounds — triangle inequality (no independence):")
    _print_upper_ci(r["BER_combined"], r["sigma_tri"])

    print(f"\n    BER_combined upper bounds — independence assumption (for comparison):")
    _print_upper_ci(r["BER_combined"], r["sigma_indep"])


def print_ber_ci_section(r):
    print(f"\n[4] Assumed QBER — upper bounds (Wald on pooled binomial)")

    print(f"\n    CHSH codec — from CHSH calibration ({r['n_E']} rounds):")
    print(f"      Point: {r['QBER_E']:.6f}   sigma: {r['sigma_QBER_E']:.6f}")
    _print_upper_ci(r["QBER_E"], r["sigma_QBER_E"])

    print(f"\n    ZZ codec — from ZZ calibration ({r['n_Q']} rounds):")
    print(f"      Point: {r['Q']:.6f}   sigma: {r['sigma_Q']:.6f}")
    _print_upper_ci(r["Q"], r["sigma_Q"])


def print_summary(r, gamma):
    print(f"\n{'='*92}")
    print(f"SUMMARY (Sep — assumed-QBER estimation)")
    print(f"{'='*92}")
    print(f"  Calibration:")
    print(f"    CHSH : n_E = {r['n_E']},  S = {r['S']:.4f} +/- {r['sigma_S']:.4f},  "
          f"QBER_E = {r['QBER_E']*100:.4f}%")
    print(f"    ZZ   : n_Q = {r['n_Q']},  Q = {r['Q']*100:.4f}%")
    print(f"    Gamma = {gamma:.10f} (used only for the legacy BER_combined)")

    print(f"\n  Assumed QBER — CHSH codec:")
    print(f"      {'CL':>6}  {'ceil(5e-4)':>12}")
    for cl, hi in _upper_bounds(r["QBER_E"], r["sigma_QBER_E"]):
        print(f"      {cl*100:>5.1f}%  {_ceil_to_0005(hi):>12.4f}")

    print(f"\n  Assumed QBER — ZZ codec:")
    print(f"      {'CL':>6}  {'ceil(5e-4)':>12}")
    for cl, hi in _upper_bounds(r["Q"], r["sigma_Q"]):
        print(f"      {cl*100:>5.1f}%  {_ceil_to_0005(hi):>12.4f}")

    print(f"\n  Next step: build the channel reliability files for each codec at")
    print(f"  the chosen assumed QBER, then run the reconciliation programs.")

    print(f"\n  CI source note:")
    print(f"    All CIs use Wald method: point + z * sigma (upper bound only).")
    print(f"    - z-scores (z_(1-a/2)): "
          f"{', '.join(f'{z:.4f}@{int(cl*1000)/10:g}%' for cl, z in CONF_LEVELS)}")
    print(f"    - Separate-IR (CHSH/ZZ codec): Wald on each calibration's pooled")
    print(f"      binomial (sigma_QBER_E, sigma_Q).")
    print(f"    - Legacy BER_combined: Wald with sigma from triangle inequality")
    print(f"      (gamma*sigma_E + (1-gamma)*sigma_Q) — no independence assumption.")
    print(f"    'assumed QBER' values are upper bounds, rounded up to 0.0005 (5e-4).")


def main():
    parser = argparse.ArgumentParser(
        description='Calibration analysis and assumed-QBER estimation (Sep).')
    parser.add_argument('--gamma', type=float,
                        default=0.018686757829462667,
                        help='CHSH round probability for the upcoming experiment')
    args = parser.parse_args()
    gamma = args.gamma

    print("=" * 92)
    print("CALIBRATION ANALYSIS — ASSUMED-QBER ESTIMATION (Sep)")
    print("=" * 92)

    r = analyze_calibrations(gamma)
    print_calibration_section(r, gamma)
    print_ber_ci_section(r)
    print_summary(r, gamma)


if __name__ == '__main__':
    main()
