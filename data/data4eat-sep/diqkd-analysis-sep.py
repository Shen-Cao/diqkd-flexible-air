#!/usr/bin/env python3
r"""
DIQKD **experimental** data analysis (Sep dataset).

The dataset is already sifted and preprocessed: the .npy files contain only
the testing and key-generation rounds retained by the sifting step (rounds
with omitted settings are already discarded), and the raw non-binary
detection records are already reduced to binary outcomes.

Computes the quantities needed for the **separate** IR pipeline:

  (1) CHSH testing rounds
      - per-subgroup joint counts, correlation E_i, and BER after optimal flip
      - S value with sigma_S
      - average BER (after optimal per-subgroup flip) with 95/99/99.9% CIs

  (2) Key-generation rounds (Z⊗Z)
      - joint counts, n_Q
      - BER with 95/99/99.9% CIs

Usage:
  python diqkd-analysis-sep.py
  python diqkd-analysis-sep.py /path/to/data_dir
"""
import os
import sys
import math
import numpy as np

# ------------------------------------------------------------------
# CI helpers (Wald + Wilson score interval)
# ------------------------------------------------------------------
# (confidence_level, z-score, label)
CONF_LEVELS = [(0.90, 1.644854), (0.95, 1.959964),
               (0.99, 2.575829), (0.999, 3.290527)]


def wald_ci(p_hat, n, z):
    """Standard Wald interval for a binomial proportion."""
    if n <= 0:
        return float("nan"), float("nan")
    sigma = math.sqrt(max(p_hat * (1.0 - p_hat) / n, 0.0))
    return p_hat - z * sigma, p_hat + z * sigma


def wilson_ci(p_hat, n, z):
    """Wilson score interval (more accurate for small n or extreme p)."""
    if n <= 0:
        return float("nan"), float("nan")
    denom = 1.0 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p_hat * (1.0 - p_hat) / n + z * z / (4 * n * n)) / denom
    return center - margin, center + margin


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------
_FILE_NAMES = [
    "alice_basis", "alice_exp_type", "alice_result",
    "bob_basis", "bob_exp_type", "bob_result",
    "snspd_click",
]


def load_data(data_dir):
    return {name: np.load(os.path.join(data_dir, f"{name}.npy"))
            for name in _FILE_NAMES}


def apply_snspd_correction(alice_basis, bob_result, snspd_click):
    """Flip Bob's outcome when Alice measures X and SNSPD channel 1 fired."""
    flip = (alice_basis == 1) & (snspd_click == 1)
    corrected = bob_result.copy()
    corrected[flip] = 1 - corrected[flip]
    return corrected


def _joint_counts(a, b):
    return np.array([
        ((a == 0) & (b == 0)).sum(),
        ((a == 0) & (b == 1)).sum(),
        ((a == 1) & (b == 0)).sum(),
        ((a == 1) & (b == 1)).sum(),
    ])


def _correlation(cells):
    total = int(cells.sum())
    if total == 0:
        return float("nan"), float("nan"), 0
    agree = cells[0] + cells[3]
    disagree = cells[1] + cells[2]
    E = (agree - disagree) / total
    sigma_E = math.sqrt(max(0.0, (1.0 - E * E) / total))
    return float(E), float(sigma_E), total


# ------------------------------------------------------------------
# Main analysis
# ------------------------------------------------------------------
def analyze(data_dir):
    data = load_data(data_dir)

    alice_basis = data["alice_basis"]
    bob_basis = data["bob_basis"]
    exp_type = data["alice_exp_type"]
    alice_result = data["alice_result"]
    bob_result_raw = data["bob_result"]
    snspd_click = data["snspd_click"]

    bob_result = apply_snspd_correction(alice_basis, bob_result_raw, snspd_click)

    # ============================================================
    # Round counts
    # ============================================================
    total = len(exp_type)
    n_chsh = int((exp_type == 0).sum())
    n_keygen = int((exp_type == 1).sum())

    # ============================================================
    # CHSH subgroups
    # ============================================================
    subgroups = []
    chsh_mask_all = (exp_type == 0)
    for ab in [0, 1]:
        for bb in [0, 1]:
            mask = chsh_mask_all & (alice_basis == ab) & (bob_basis == bb)
            cells = _joint_counts(alice_result[mask], bob_result[mask])
            E, sigma_E, n = _correlation(cells)
            # Per-subgroup BER after optimal flip:
            #   if E > 0: errors = disagree (n01 + n10), flip ON
            #   if E < 0: errors = agree   (n00 + n11), flip OFF
            if E > 0:
                errors = int(cells[1] + cells[2])
                flip = True
            else:
                errors = int(cells[0] + cells[3])
                flip = False
            subgroups.append({
                "a_basis": ab, "b_basis": bb,
                "cells": cells, "n": n,
                "E": E, "sigma_E": sigma_E,
                "errors_after_flip": errors,
                "ber_after_flip": errors / n if n > 0 else float("nan"),
                "flip": flip,
            })

    # Pooled CHSH BER (optimal flip)
    chsh_total_errors = sum(sg["errors_after_flip"] for sg in subgroups)
    chsh_total_n = sum(sg["n"] for sg in subgroups)
    chsh_ber = chsh_total_errors / chsh_total_n

    # S value
    S = sum(abs(sg["E"]) for sg in subgroups)
    var_S = sum((1.0 - sg["E"] ** 2) / sg["n"]
                for sg in subgroups if sg["n"] > 0)
    sigma_S = math.sqrt(var_S)

    # ============================================================
    # Key-gen rounds (Z⊗Z)
    # ============================================================
    kg_mask = (exp_type == 1) & (bob_basis == 2)
    kg_cells = _joint_counts(alice_result[kg_mask], bob_result[kg_mask])
    n_kg = int(kg_cells.sum())
    # QBER convention: P(a == b) on raw outcomes (singlet anti-corr).
    # BER convention for IR: same numerically (a != b'), where b' = 1 - b
    # (singlet flip). So BER = P(a == b) = (n00 + n11) / n.
    kg_ber = float((kg_cells[0] + kg_cells[3]) / n_kg) if n_kg > 0 else float("nan")
    # Also expose E_keygen for diagnostic
    agree_kg = int(kg_cells[0] + kg_cells[3])
    disagree_kg = int(kg_cells[1] + kg_cells[2])
    E_kg = (agree_kg - disagree_kg) / n_kg if n_kg > 0 else float("nan")

    return {
        "data_dir": data_dir,
        "total": total,
        "n_chsh": n_chsh,
        "n_keygen": n_keygen,
        "subgroups": subgroups,
        "chsh_total_errors": chsh_total_errors,
        "chsh_total_n": chsh_total_n,
        "chsh_ber": chsh_ber,
        "S": S,
        "sigma_S": sigma_S,
        "kg_cells": kg_cells,
        "n_kg": n_kg,
        "kg_ber": kg_ber,
        "E_kg": E_kg,
    }


# ------------------------------------------------------------------
# Reporting
# ------------------------------------------------------------------
def _fmt_pct(x, digits=4):
    return f"{x * 100:.{digits}f}%"


def _print_ci_table(p_hat, n, header):
    print(f"    {header}: {_fmt_pct(p_hat)}  (n = {n:,})")
    print(f"      {'CL':>6}  {'Wald lower':>14}  {'Wald upper':>14}"
          f"  {'Wilson lower':>14}  {'Wilson upper':>14}"
          f"  {'Ceil(Wald,5e-4)':>16}")
    for cl, z in CONF_LEVELS:
        lo_w, hi_w = wald_ci(p_hat, n, z)
        lo_wl, hi_wl = wilson_ci(p_hat, n, z)
        ceil_hi = math.ceil(hi_w * 2000) / 2000  # rounded up to 5e-4
        print(f"      {cl * 100:>5.1f}%  {_fmt_pct(lo_w):>14}  {_fmt_pct(hi_w):>14}"
              f"  {_fmt_pct(lo_wl):>14}  {_fmt_pct(hi_wl):>14}"
              f"  {ceil_hi:>16.4f}")


def print_report(r):
    print("=" * 92)
    print("DIQKD Experimental Data Analysis (Sep dataset)")
    print("=" * 92)
    print(f"  Data directory: {r['data_dir']}")

    # ============================================================
    # [1] Round counts
    # ============================================================
    print(f"\n[1] Round counts")
    print(f"    Total rounds              : {r['total']:,}")
    print(f"    CHSH testing (exp_type=0) : {r['n_chsh']:,}")
    print(f"    Key generation (exp_type=1): {r['n_keygen']:,}")

    # ============================================================
    # [2] CHSH subgroups
    # ============================================================
    print(f"\n[2] CHSH testing rounds — per-subgroup breakdown")
    labels = ["(a=Z, b=Z+X)", "(a=Z, b=Z-X)", "(a=X, b=Z+X)", "(a=X, b=Z-X)"]
    print(f"    {'Subgroup':<18}{'cells [n00,n01,n10,n11]':<32}"
          f"{'n':>7}{'E':>10}{'sigma_E':>10}"
          f"{'Flip?':>8}{'BER':>10}")
    print("    " + "-" * 92)
    for sg, lbl in zip(r["subgroups"], labels):
        c = sg["cells"]
        cells_str = f"[{c[0]:>5},{c[1]:>5},{c[2]:>5},{c[3]:>5}]"
        flip_str = "YES" if sg["flip"] else "NO"
        print(f"    {lbl:<18}{cells_str:<32}{sg['n']:>7}"
              f"{sg['E']:>+10.4f}{sg['sigma_E']:>10.4f}"
              f"{flip_str:>8}{sg['ber_after_flip'] * 100:>9.4f}%")

    abs_E_str = " + ".join(f"{abs(sg['E']):.4f}" for sg in r["subgroups"])
    print(f"\n    S = {abs_E_str}")
    print(f"      = {r['S']:.4f} +/- {r['sigma_S']:.4f}")

    print(f"\n    CHSH pooled BER (after optimal per-subgroup flip):")
    print(f"      Total errors = {r['chsh_total_errors']:,} / {r['chsh_total_n']:,}")

    _print_ci_table(r["chsh_ber"], r["chsh_total_n"],
                    header="CHSH BER (pooled, optimal flip)")

    # ============================================================
    # [3] Key-gen rounds
    # ============================================================
    print(f"\n[3] Key generation rounds (Z⊗Z)")
    c = r["kg_cells"]
    print(f"    cells [n00,n01,n10,n11] = [{c[0]:>7},{c[1]:>7},{c[2]:>7},{c[3]:>7}]")
    print(f"    n_Q = {r['n_kg']:,}")
    print(f"    E_keygen = {r['E_kg']:+.6f}")

    _print_ci_table(r["kg_ber"], r["n_kg"],
                    header="KeyGen BER (P(a==b) raw, = IR BER after singlet flip)")

    # ============================================================
    # [4] CI summary (compact)
    # ============================================================
    print(f"\n[4] Compact CI summary (Wald, upper bounds)")
    print(f"    {'Quantity':<28}{'Point':>11}{'90% up':>11}"
          f"{'95% up':>11}{'99% up':>11}{'99.9% up':>12}")
    print("    " + "-" * 84)

    def _row(label, p, n):
        cells = [label, _fmt_pct(p, 3)]
        for cl, z in CONF_LEVELS:
            _, hi = wald_ci(p, n, z)
            cells.append(_fmt_pct(hi, 3))
        fmt = f"    {cells[0]:<28}{cells[1]:>11}"
        for c in cells[2:]:
            fmt += f"{c:>11}" if c != cells[-1] else f"{c:>12}"
        print(fmt)

    _row("CHSH BER (optimal flip)", r["chsh_ber"], r["chsh_total_n"])
    _row("KeyGen BER", r["kg_ber"], r["n_kg"])

    print(f"\n  Notes:")
    print(f"  - CHSH BER uses per-subgroup optimal flip (E>0 → flip).")
    print(f"  - KeyGen BER convention: P(a==b) on raw outcomes, equal to IR BER")
    print(f"    after singlet-flip (b' = 1 - b).")
    print(f"  - All CIs are 2-sided Wald on the pooled binomial.")
    print(f"  - For polar-code construction, pick the CI upper bound matching")
    print(f"    your target confidence level (e.g. 99.9% upper for conservative")
    print(f"    design) and round up to the nearest 0.0005 (5e-4).")


def main(data_dir=None):
    if data_dir is None:
        data_dir = os.path.dirname(os.path.abspath(__file__))
    result = analyze(data_dir)
    print_report(result)
    return result


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else None
    main(data_dir)
