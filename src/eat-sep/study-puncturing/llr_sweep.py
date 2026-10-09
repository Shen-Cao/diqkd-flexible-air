#!/usr/bin/env python3
"""LLR-magnitude sweep at the punctured positions (synthetic, short length).

Shows that no finite amplification of the known-zero padding at the punctured
positions can substitute for the missing hard constraint.
"""
import synthetic_study as ss


class A:
    pass


A.fbgen = ss.DEFAULT_FBGEN
A.frames = 2000

print("| LLR at punctured positions | key FER |")
print("|---|---|")
for tag, mode in [("0 (erasure)", "erase"), ("1x (raw)", "amp:0"),
                  ("17x", "amplify"), ("1e3x", "amp:1000"),
                  ("1e6x", "amp:1000000")]:
    kf, lf, ub, K = ss.run_case(A, 10, "punct", 358, 0.12, 1e-2, 8, mode, A.frames)
    print("| %s | **%.4f** |" % (tag, kf), flush=True)