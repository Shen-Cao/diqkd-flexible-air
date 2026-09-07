#!/usr/bin/env python3
"""
Generate channel indices sorted by channel reliability.

Reads a *_pe.txt file (tab-separated ``index\\tP_e`` lines, one per channel)
and emits a *.pc file whose 4th line is the channel ordering sorted by
ascending P_e (most reliable first). The output matches the format of the
existing ``conf/polar/sp-bsc/*.pc`` files:

    line 1: N               (block length, inferred from the input)
    line 2: channel type    (e.g. ``sp-bsc``)
    line 3: BER             (e.g. ``0.038``)
    line 4: i_0 i_1 ...     (channel indices, space-separated, most reliable first)

Example
-------
    python data/pe2idx.py \\
        --input  conf/polar/sp-bsc/frozen_bits_mu=40_N=20_p=0.0365_shortened=187329_punctured=0.txt \\
        --output conf/polar/sp-bsc/n20_sp-bsc_ber0.0365_mu40_shortened187329_punctured0.pc \\
        --channel-type sp-bsc \\
        --ber 0.0365
"""

import argparse
import os

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a *.pc reliability-ordering file from a *_pe.txt file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to the input *_pe.txt file (tab-separated 'index\\tP_e').",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Path to the output *.pc file.",
    )
    parser.add_argument(
        "-t", "--channel-type",
        required=True,
        help="Channel type written to the .pc header (e.g. 'sp-bsc').",
    )
    parser.add_argument(
        "-p", "--ber",
        required=True,
        type=float,
        help="Bit error rate written to the .pc header (e.g. 0.047).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    # Read the P_e column (column index 1). Each row is "idx\tP_e".
    pe_v = np.loadtxt(args.input, delimiter="\t", usecols=1, dtype=np.float64)
    N = pe_v.size

    # Ascending sort: the first index in the output is the one with the lowest
    # P_e, i.e. the most reliable channel. Matches the ordering of the .pc
    # files in conf/polar/sp-bsc/ (consumed by generate_frozen_position_from_file,
    # which freezes the *least* reliable channels, i.e. the tail of this vector).
    sorted_idx = np.argsort(pe_v)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # The .pc layout is: 3 header lines (N, channel type, BER) followed by one
    # line of space-separated indices. np.savetxt with delimiter='' and
    # newline=' ' emits "i_0 i_1 ... " (single trailing space, no trailing
    # newline) — byte-identical to the .pc files shipped in conf/polar/sp-bsc/.
    with open(args.output, "w") as f:
        f.write(f"{N}\n")
        f.write(f"{args.channel_type}\n")
        f.write(f"{args.ber}\n")
        np.savetxt(f, sorted_idx, fmt="%d", delimiter="", newline=" ")

    print(f"Wrote {args.output} (N={N}, channel={args.channel_type}, ber={args.ber})")


if __name__ == "__main__":
    main()
