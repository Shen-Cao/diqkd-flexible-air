#!/usr/bin/env python3
"""Synthetic study: BGL puncturing vs shortening inside the AIR decoding frame.

Here we use synthetic data produced with a BSC over random messages at short
code lengthes. The polar-code reliability tables are generated on the fly with
polar-fbgen-rust.

Usage:
    python synthetic_study.py [--fbgen PATH] [--frames N] [--out results.md]
"""
import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np


def _default_fbgen():
    found = shutil.which("polar_fbgen")
    if found:
        return found
    return os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "..", "tmp", "polar-fbgen-rust", "target", "release", "polar_fbgen"))


DEFAULT_FBGEN = os.environ.get("POLAR_FBGEN", _default_fbgen())
PY_AFF3CT_LIB = os.environ.get(
    "PY_AFF3CT_LIB",
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "..", "..", "..", "submodules", "py_aff3ct", "build", "lib")),
)
# make the bindings importable for this script and for anything that imports it
sys.path.insert(0, PY_AFF3CT_LIB)


def reverse_bits(i, nb):
    r = 0
    for _ in range(nb):
        r = (r << 1) | (i & 1)
        i >>= 1
    return r


def bgl_positions(N, count, kind):
    n = int(round(math.log2(N)))
    if kind == "punct":
        return np.array([reverse_bits(i, n) for i in range(0, count)], dtype=np.int64)
    return np.array([reverse_bits(i, n) for i in range(N - count, N)], dtype=np.int64)


def make_tables(fbgen, n, kind, count, ber):
    out = tempfile.mkdtemp(prefix="study_")
    flag = "--num-punctured" if kind == "punct" else "--num-shortened"
    cmd = [fbgen, "--method", "batch", "-N", str(n), "--mu", "40",
           "-p", str(ber), flag, str(count), "-o", out]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    pe_file = [f for f in os.listdir(out) if f.startswith("frozen_bits")][0]
    pe = np.loadtxt(os.path.join(out, pe_file), usecols=1)
    pos_file = [f for f in os.listdir(out)
                if f.startswith(("punctured_positions", "shortened_positions"))][0]
    pos = np.array([int(l) for l in open(os.path.join(out, pos_file))
                    if l.strip() and not l.startswith("#")], dtype=np.int64)
    return pe, pos


def run_case(args, n, kind, trim, ber, ub_target, scl, llr_mode, frames, seed=1234):
    """Return (key_fer, leaf_fer, union_bound, K) or None."""
    from py_aff3ct.module.encoder import Encoder_polar
    from py_aff3ct.module.decoder import Decoder_polar_SCL_naive

    N = 1 << n
    pe, pos = make_tables(args.fbgen, n, kind, trim, ber)
    sp = np.zeros(N, bool)
    sp[pos] = True
    non_sp = ~sp

    order = np.argsort(pe)
    by_rel = [i for i in order[::-1] if not sp[i]]
    k = 0
    fv = np.zeros(N, bool)
    while k < len(by_rel):
        fv = np.zeros(N, bool)
        if kind == "short":
            fv[pos] = True
        fv[by_rel[:k]] = True
        ub = float(pe[(~fv) & non_sp].sum())
        if ub <= ub_target:
            break
        k += 1
    K = int((~fv).sum())
    if K < 1:
        return None

    enc = Encoder_polar(K, N, fv)

    def apply_G(v):
        o = np.zeros((1, N), np.int32)
        enc["light_encode ::U_N"].bind(np.asarray(v, dtype=np.int32).reshape(1, -1))
        enc["light_encode ::X_N"].bind(o)
        enc["light_encode"].exec()
        return o.flatten()

    rng = np.random.default_rng(seed)
    base = float(np.log((1 - ber) / ber))
    key_err = 0
    leaf_err = 0
    for _ in range(frames):
        U = np.zeros(N, np.int32)
        U[non_sp] = rng.integers(0, 2, size=int(non_sp.sum()))
        X = apply_G(U)
        e = (rng.random(N) < ber).astype(np.int32)
        e[sp] = 0
        Y = ((1 - 2 * (U ^ e)).astype(np.float32)) * np.float32(base)
        if llr_mode == "erase":
            Y[sp] = 0.0
        elif llr_mode == "amplify" or llr_mode.startswith("amp:"):
            factor = 16.0 if llr_mode == "amplify" else float(llr_mode.split(":")[1])
            Y = Y + (sp.astype(np.float32) * Y) * np.float32(factor)
        F = np.zeros(N, np.int32)
        F[fv] = X[fv]
        d = Decoder_polar_SCL_naive(K, N, scl, fv)
        d.set_frozen_bits(fv)
        V = np.zeros((1, N), np.int32)
        st = np.ones((1, 1), np.int32)
        d["decode_siho_cw_flexible_frozen ::Y_N"].bind(Y.reshape(1, -1))
        d["decode_siho_cw_flexible_frozen ::F_N"].bind(F.reshape(1, -1))
        d["decode_siho_cw_flexible_frozen ::V_N"].bind(V)
        d["decode_siho_cw_flexible_frozen ::status"].bind(st)
        d["decode_siho_cw_flexible_frozen"].exec()
        V = V.flatten()
        leaf_err += int(not np.array_equal(V, X))
        key_err += int(not np.array_equal(apply_G(V)[non_sp], U[non_sp]))
    return key_err / frames, leaf_err / frames, ub, K


PLAN = [
    (9,  "punct", 0.40, 0.12, 1e-2, 8,  "amplify"),
    (9,  "punct", 0.40, 0.12, 1e-2, 8,  "erase"),
    (9,  "punct", 0.40, 0.12, 1e-2, 32, "erase"),
    (9,  "short", 0.40, 0.12, 1e-2, 8,  "amplify"),
    (9,  "short", 0.40, 0.12, 1e-2, 32, "amplify"),
    (10, "punct", 0.35, 0.12, 1e-2, 8,  "amplify"),
    (10, "punct", 0.35, 0.12, 1e-2, 8,  "erase"),
    (10, "short", 0.35, 0.12, 1e-2, 8,  "amplify"),
    (10, "punct", 0.35, 0.12, 1e-3, 8,  "erase"),
    (10, "punct", 0.35, 0.12, 1e-3, 32, "erase"),
    (10, "short", 0.35, 0.12, 1e-3, 8,  "amplify"),
    (10, "short", 0.35, 0.12, 1e-3, 32, "amplify"),
]


def main():
    sys.path.insert(0, PY_AFF3CT_LIB)
    rows = []
    for (n, kind, frac, ber, ubt, scl, mode) in PLAN:
        trim = int(round(frac * (1 << n)))
        res = run_case(ARGS, n, kind, trim, ber, ubt, scl, mode, ARGS.frames)
        if res is None:
            print("skip", n, kind, ubt)
            continue
        kf, lf, ub, K = res
        rows.append((n, 1 << n, kind, trim, ber, ubt, scl, mode, K, ub, kf, lf))
        print("n=%2d N=%4d %-5s trim=%4d ber=%.2f scl=%2d %-8s K=%4d ub=%.3g keyFER=%.4f leafFER=%.4f"
              % (n, 1 << n, kind, trim, ber, scl, mode, K, ub, kf, lf), flush=True)
    with open(ARGS.out, "w") as f:
        f.write("| n | N | set | trim | BER | ub target | SCL | LLR at trimmed positions | K | union bound | key FER | leaf (codeword) FER |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write("| %d | %d | %s | %d | %.2f | %.0e | %d | %s | %d | %.3g | **%.4f** | %.4f |\n" % r)
    print("wrote", ARGS.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fbgen", default=DEFAULT_FBGEN)
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--out", default="results.md")
    ARGS = ap.parse_args()
    main()
