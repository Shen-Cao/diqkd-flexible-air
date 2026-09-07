#!/usr/bin/env python3
# r"""
# AIR pipeline for **ZZ key-gen rounds** of the separate-IR DIQKD protocol (Sep).
#
# Reconciles Alice's Z⊗Z key-generation bits with Bob via polar-code AIR,
# using a pure-shortening polar code (BGL construction) at
# N = 2^20 = 1048576 to handle the low-BER (~3.4%) regime of key-gen rounds.
# (Puncturing is still supported by the code path, but it is incompatible with
# this decoder -- see src/eat-sep/study-puncturing/README.md.)
#
# Differences vs. ``ir-raw-chsh.py``:
#   1. Data source: only ZZ key-gen rounds (exp_type==1, bob_basis==2),
#      ~861k pairs.
#   2. Polar code: N = 2^20 = 1048576 (much larger). Memory ~4.5 GB per
#      decoder at SCL=8.
#   3. Decoder SCL default = 8 (capped due to RAM at N=2^20).
#   4. Default BER lower (~0.038 vs ~0.18).
#   5. Trimming configuration: pure shortening (S=187329 + P=0, default;
#      S + P = N - L). Mixed trims (e.g. S=150000 + P=37170) remain
#      expressible, but the decoder constrains only the shortened padding;
#      punctured leaves stay unconstrained and the study in
#      study-puncturing/README.md shows that variant is incompatible with
#      the current IR protocol.
#
# Memory management:
#   - Each Decoder_polar_SCL_naive_CA at N=2^20, SCL=8 uses ~4.5 GB.
#   - We cannot hold multiple decoders at once. Each stage's decoder is
#     constructed on demand, used once, then freed (del + gc.collect)
#     before the next stage — same pattern as ``ir-raw-chsh.py``.
#
# Usage:
#   python ir-raw-keygen.py [ASSUMED_BER] [S] [P] [SCL_LEN]
#
# Examples:
#   python ir-raw-keygen.py                              # defaults: BER=0.0365, S=187329, P=0, L=8
#   python ir-raw-keygen.py --eff-stages 1.04 1.05 1.06  # focus on specific efficiency stages
#   python ir-raw-keygen.py --save-result --eff-stages 1.05 1.10  # save Alice/Bob result .npy
#   python ir-raw-keygen.py 0.038 187329 0 8             # other BERs need matching .pc files (see README)
# """
import sys
import os
import time
import gc
import math
import argparse
from datetime import datetime
from copy import deepcopy

sys.path.insert(0, '../../submodules/py_aff3ct/build/lib')
import numpy as np
import py_aff3ct

sys.path.insert(0, '../../src/')
from utils import (
    generate_shortened_from_file, generate_punctured_from_file,
    generate_shortened_BGL, generate_punctured_BGL,
    generate_frozen_position_from_file,
    num_frozen_bits_from_union_bound_from_file,
    llr_calculater_bsc, mask_fillor, extractor,
)
from py_aff3ct.module.encoder import Encoder_polar
from py_aff3ct.module.decoder import Decoder_polar_SCL_naive_CA
from py_aff3ct.module.crc import CRC_polynomial

# ============================================================
# Configuration
# ============================================================
DATA_DIR = '../../data/data4eat-sep/'
PC_DIR   = '../../conf/polar/sp-bsc/'
LOG_DIR  = '../../log/eat-diqkd-sep/'

DEFAULT_BER    = 0.0365
DEFAULT_S      = 187329
DEFAULT_P      = 0
DEFAULT_SCL    = 8
CRC_SIZE       = 32

# Configured AIR efficiency stages (same schedule as the demo notebooks):
# f = 1.05 and f = 1.10 vs. the assumed ideal, plus a final stage sized by
# the union bound on the frozen_bits PE file targeting a failure rate of
# 1e-5 (assembled after the sp positions are loaded below).
CONFIGURED_EFFICIENCY_STAGE = np.array([1.05, 1.10], dtype=np.float64)

shannon_H2 = lambda x: -x * np.log2(x) - (1 - x) * np.log2(1 - x) if 0 < x < 1 else 0.0

# ============================================================
# CLI
# ============================================================
parser = argparse.ArgumentParser(
    description='AIR pipeline for ZZ key-gen rounds (separate IR).',
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument('assumed_ber', type=float, nargs='?', default=DEFAULT_BER,
                    help=f'Assumed BER (default: {DEFAULT_BER})')
parser.add_argument('S', type=int, nargs='?', default=DEFAULT_S,
                    help=f'Number of shortened bits (default: {DEFAULT_S})')
parser.add_argument('P', type=int, nargs='?', default=DEFAULT_P,
                    help=f'Number of punctured bits (default: {DEFAULT_P})')
parser.add_argument('scl_len', type=int, nargs='?', default=DEFAULT_SCL,
                    help=f'SCL decoder list size (default: {DEFAULT_SCL}, max recommended 8 at N=2^20)')
parser.add_argument('--eff-stages', type=float, nargs='+', default=None,
                    help='Override the efficiency stages (default: f=1.05, 1.10 plus '
                         'a union-bound final stage at failure rate 1e-5). '
                         'Example: --eff-stages 1.04 1.05 1.06')
parser.add_argument('--save-result', action='store_true',
                    help='Save the reconciliation result as .npy files: Alice sifted '
                         'key and Bob corrected key (effective, non-sp positions only). '
                         'Default: off.')
args = parser.parse_args()

ASSUMED_BER = args.assumed_ber
S_BITS = args.S
P_BITS = args.P
SCL_LEN = args.scl_len
SAVE_RESULT = args.save_result

# ============================================================
# Load ZZ keygen data and apply 3-step flip pipeline
# ============================================================
# For ZZ keygen rounds, only Step 1 (SNSPD correction) and Step 3 (AIR flip)
# apply. Step 2 (per-subgroup CHSH flip) is for Bell-test rounds only.
alice_basis    = np.load(os.path.join(DATA_DIR, 'alice_basis.npy'))
alice_exp_type = np.load(os.path.join(DATA_DIR, 'alice_exp_type.npy'))
alice_result   = np.load(os.path.join(DATA_DIR, 'alice_result.npy')).astype(np.int8)
bob_basis      = np.load(os.path.join(DATA_DIR, 'bob_basis.npy'))
bob_exp_type   = np.load(os.path.join(DATA_DIR, 'bob_exp_type.npy'))
bob_result     = np.load(os.path.join(DATA_DIR, 'bob_result.npy')).astype(np.int8)
snspd_click    = np.load(os.path.join(DATA_DIR, 'snspd_click.npy'))

assert np.array_equal(alice_exp_type, bob_exp_type), \
    "Alice and Bob exp_type mismatch."

# Keep only ZZ keygen rounds (exp_type == 1, bob_basis == 2)
zz_mask = (alice_exp_type == 1) & (bob_basis == 2)
alice_basis_zz  = alice_basis[zz_mask]
alice_result_zz = alice_result[zz_mask]
bob_result_zz   = bob_result[zz_mask]
snspd_click_zz  = snspd_click[zz_mask]

num_kg = int(zz_mask.sum())
print(f"Loaded {num_kg} ZZ keygen rounds from {DATA_DIR}")

# Step 1: SNSPD correction (applies to all rounds uniformly).
# In keygen rounds, Alice measures in Z basis (alice_basis in {0, 2}),
# so alice_basis==1 should never occur here, but the correction is a no-op
# in that case anyway. Apply defensively.
flip_snspd = (alice_basis_zz == 1) & (snspd_click_zz == 1)
n_flipped_snspd = int(flip_snspd.sum())
if n_flipped_snspd > 0:
    bob_result_zz = bob_result_zz.copy()
    bob_result_zz[flip_snspd] = 1 - bob_result_zz[flip_snspd]
    print(f"  SNSPD correction flipped {n_flipped_snspd} bits")

# Step 2 (CHSH flip): SKIPPED for keygen rounds.

# Step 3: AIR flip on Alice (a -> 1-a) to convert anti-correlation to BSC convention.
alice_result_zz = (1 - alice_result_zz).astype(np.int8)

# Compute actual noise rate
ir_ber = float(np.logical_xor(alice_result_zz, bob_result_zz).sum() / num_kg)
print(f"ZZ keygen BER after 3-step flip: {ir_ber:.6f}  ({ir_ber*100:.4f}%)")

# ============================================================
# Polar code parameters
# ============================================================
L_DATA = num_kg                    # 861247
N = 1 << 20                        # 1048576
assert N >= L_DATA + S_BITS + P_BITS, \
    f"N={N} too small for L+S+P = {L_DATA}+{S_BITS}+{P_BITS}"
assert S_BITS + P_BITS == N - L_DATA, \
    f"S+P must equal N-L: {S_BITS}+{P_BITS} != {N}-{L_DATA}"

n_pow = 20
print(f"\nPolar code: N = 2^{n_pow} = {N}, L = {L_DATA}, "
      f"S = {S_BITS}, P = {P_BITS}, SCL = {SCL_LEN}")

# ============================================================
# Load shortened / punctured positions
# ============================================================
sp_file_S = os.path.join(PC_DIR, f'shortened_positions_N={n_pow}_S={S_BITS}.txt')
sp_file_P = os.path.join(PC_DIR, f'punctured_positions_N={n_pow}_P={P_BITS}.txt')

if os.path.exists(sp_file_S):
    shortened_idx = generate_shortened_from_file(sp_file_S)
    print(f"Shortened positions loaded from file: {len(shortened_idx)} positions")
else:
    shortened_idx = generate_shortened_BGL(N, S_BITS)
    print(f"Shortened positions generated via BGL: {len(shortened_idx)} positions")

if os.path.exists(sp_file_P):
    punctured_idx = generate_punctured_from_file(sp_file_P)
    print(f"Punctured positions loaded from file: {len(punctured_idx)} positions")
else:
    punctured_idx = generate_punctured_BGL(N, P_BITS)
    print(f"Punctured positions generated via BGL: {len(punctured_idx)} positions")

# Sanity: disjoint
overlap = set(shortened_idx.tolist()) & set(punctured_idx.tolist())
assert len(overlap) == 0, f"Shortened and punctured sets overlap at {len(overlap)} positions"

shortened_vec = np.zeros(N, dtype=bool)
shortened_vec[shortened_idx] = True
punctured_vec = np.zeros(N, dtype=bool)
punctured_vec[punctured_idx] = True
sp_vec = shortened_vec | punctured_vec
sp_idx = np.concatenate([shortened_idx, punctured_idx])
sp_set = set(int(x) for x in sp_idx)

non_sp_mask = (~sp_vec).astype(np.int32).reshape((1, -1))
non_sp_1d = np.where(non_sp_mask.flatten() == 1)[0]
assert len(non_sp_1d) == L_DATA, \
    f"non-sp count {len(non_sp_1d)} != L_DATA {L_DATA}"

print(f"  Total sp positions: {len(sp_idx)} ({S_BITS} short + {P_BITS} punct)")
print(f"  Transmitted positions: {len(non_sp_1d)} (== L_DATA)")

# --- Efficiency stages ---
if args.eff_stages is not None:
    EFFICIENCY_STAGE = np.array(args.eff_stages, dtype=np.float64)
else:
    # Same schedule as the demo notebooks: the configured stages plus a
    # final union-bound stage targeting a failure rate of 1e-5.
    pe_file = os.path.join(
        PC_DIR,
        f'frozen_bits_mu=40_N={n_pow}_p={ASSUMED_BER}_shortened={S_BITS}_punctured={P_BITS}.txt',
    )
    kf_ub = num_frozen_bits_from_union_bound_from_file(pe_file, sp_idx, 1e-5)
    eff_ub = kf_ub / (L_DATA * shannon_H2(ASSUMED_BER))
    EFFICIENCY_STAGE = np.concatenate([CONFIGURED_EFFICIENCY_STAGE, [eff_ub]])
    print(f"Union-bound final stage: f = {eff_ub:.4f} "
          f"(K_frozen = {kf_ub}, eps = 1e-5)")

# ============================================================
# Known-zero padding mask (fed to the decoder's set_known_bits task)
# ============================================================
# The padded sp positions hold the public constant 0 (for this pure-shortened
# codec sp == shortened). The decoder amplifies the LLR at these positions
# internally (same x17 factor the external shortener used to apply).
known_bits_mask = sp_vec.astype(np.int32).reshape((1, -1))

# ============================================================
# Pad Alice and Bob data to length N
# ============================================================
padder_alice = mask_fillor(L_DATA, N, dtype=np.int32)
padder_bob   = mask_fillor(L_DATA, N, dtype=np.int32)
ext_crc      = extractor(N + CRC_SIZE, N, N + CRC_SIZE)

fixed_alice = alice_result_zz.reshape((1, -1)).astype(np.int32)
fixed_bob   = bob_result_zz.reshape((1, -1)).astype(np.int32)

padded_alice = np.zeros((1, N), dtype=np.int32)
padder_alice["fill ::output"].bind(padded_alice)
padder_alice["fill ::mask"].bind(non_sp_mask)
padder_alice["fill ::input"].bind(fixed_alice)
padder_alice["fill"].exec()

padded_bob = np.zeros((1, N), dtype=np.int32)
padder_bob["fill ::output"].bind(padded_bob)
padder_bob["fill ::mask"].bind(non_sp_mask)
padder_bob["fill ::input"].bind(fixed_bob)
padder_bob["fill"].exec()

pad_xor = np.logical_xor(padded_alice, padded_bob).flatten()
pad_ber = float(pad_xor[non_sp_1d].sum()) / L_DATA
print(f"Sanity: padded non-sp BER = {pad_ber:.6f}  (should match ir_ber = {ir_ber:.6f})")
assert abs(pad_ber - ir_ber) < 1e-9, "Padded BER mismatch!"

ideal_frozen_real = L_DATA * shannon_H2(ir_ber)

# ============================================================
# Logger
# ============================================================
os.makedirs(LOG_DIR, exist_ok=True)
start_time = datetime.now()
log_file = os.path.join(
    LOG_DIR,
    f'keygen_ir_ber{ASSUMED_BER}_S{S_BITS}_P{P_BITS}_L{SCL_LEN}_'
    f'{start_time.strftime("%Y%m%d_%H%M%S")}.txt',
)

class _Logger:
    def __init__(self, lf):
        self.log_file = lf
        self.terminal = sys.stdout

    def write(self, msg):
        self.terminal.write(msg)
        with open(self.log_file, 'a') as f:
            f.write(msg)

    def flush(self):
        self.terminal.flush()
        with open(self.log_file, 'a') as f:
            f.flush()

sys.stdout = _Logger(log_file)
print(f"\nLog file: {log_file}")
print(f"Started at: {start_time}")
print(f"\n=== ZZ KeyGen AIR Pipeline (Sep, separate IR) ===")
print(f"  Assumed BER    : {ASSUMED_BER}")
print(f"  N / L / S / P  : {N} / {L_DATA} / {S_BITS} / {P_BITS}")
print(f"  SCL length     : {SCL_LEN}")
print(f"  CRC size       : {CRC_SIZE}")
print(f"  Real BER       : {ir_ber:.6f}  H2={shannon_H2(ir_ber):.6f}"
      f"  Shannon={ideal_frozen_real:.1f}")

# ============================================================
# AIR pipeline (per-assumed-BER; mirrors ir-raw-chsh.py structure)
# ============================================================
def run_air_pipeline(assumed_q_ber):
    print("\n" + "#" * 70)
    print(f"# ASSUMED BER = {assumed_q_ber}, S={S_BITS}, P={P_BITS}, SCL={SCL_LEN}")
    print("#" * 70)

    # --- Load reliability file (.pc) ---
    fb_file = os.path.join(
        PC_DIR,
        f'n{n_pow}_sp-bsc_ber{assumed_q_ber}_mu40_shortened{S_BITS}_punctured{P_BITS}.pc',
    )
    if not os.path.exists(fb_file):
        print(f"  *** .pc file not found: {fb_file}")
        return None

    print(f"  Loading polar code: {fb_file}")
    with open(fb_file, 'r') as f:
        reliability_sorted_idx = np.array(
            [int(x) for x in f.readlines()[3].split()], dtype=np.int64
        )
    assert reliability_sorted_idx.shape[0] == N

    assumed_ideal_frozen = L_DATA * shannon_H2(assumed_q_ber)
    crc_gain = CRC_SIZE / assumed_ideal_frozen

    eff_stage_cfg = EFFICIENCY_STAGE - crc_gain
    eff_frozen = np.asarray(eff_stage_cfg * assumed_ideal_frozen, dtype=np.int64)
    info_bits = N - eff_frozen - S_BITS - P_BITS
    # Decoder information positions: AIR information + freely-decoded punctured leaves
    dec_K = N - eff_frozen - S_BITS
    assert (info_bits > 0).all(), f"info_bits must be > 0, got {info_bits}"
    assert np.all(np.diff(eff_frozen) > 0), "eff_frozen must be strictly increasing"

    print(f"  H2(assumed) = {shannon_H2(assumed_q_ber):.6f}, "
          f"L*H2 = {assumed_ideal_frozen:.1f}")
    print(f"  CRC gain    : {crc_gain:.6f}")
    print(f"  Eff stages  : {EFFICIENCY_STAGE.tolist()}")
    print(f"  Eff frozen  : {eff_frozen.tolist()}")
    print(f"  Info bits   : {info_bits.tolist()}")

    # --- Build frozen-bit vectors per stage ---
    frozen_vecs = []
    eff_frozen_masks = []
    for nf in eff_frozen:
        N_pc, _, eff_idx = generate_frozen_position_from_file(
            fb_file, int(nf), sp_idx, verbose=False)
        assert N_pc == N
        # Decoder frozen = shortened + effective only; punctured leaves free
        fv = shortened_vec.copy()
        fv[eff_idx] = True
        ev = np.zeros(N, dtype=bool)
        ev[eff_idx] = True
        frozen_vecs.append(fv)
        eff_frozen_masks.append(ev.astype(np.int32).reshape((1, -1)))

    # --- Build per-assumed-BER modules (encoder + LLR) ---
    enc = Encoder_polar(int(dec_K[0]), N, frozen_vecs[0])
    enc.set_frozen_bits(frozen_vecs[0])
    crc_alice = CRC_polynomial(N, '0x04C11DB7', CRC_SIZE)
    llr_bsc   = llr_calculater_bsc(assumed_q_ber, N)

    # --- Alice: encode + CRC tag ---
    encoded_alice = np.zeros_like(padded_alice)
    enc["light_encode ::U_N"].bind(padded_alice)
    enc["light_encode ::X_N"].bind(encoded_alice)
    enc["light_encode"].exec()

    # Sanity: shortened positions in encoded_alice should be 0 (BGL property)
    if S_BITS > 0:
        n_short_nonzero = int(encoded_alice.flatten()[shortened_idx].sum())
        print(f"  Sanity: shortened positions in encoded_alice, "
              f"sum = {n_short_nonzero} (should be 0 by BGL design)")

    enc_crc_buf = np.zeros((1, N + CRC_SIZE), dtype=np.int32)
    crc_alice["build ::U_K1"].bind(encoded_alice)
    crc_alice["build ::U_K2"].bind(enc_crc_buf)
    crc_alice["build"].exec()
    crc_tag = np.zeros((1, CRC_SIZE), dtype=np.int32)
    ext_crc["extract ::input"].bind(enc_crc_buf)
    ext_crc["extract ::output"].bind(crc_tag)
    ext_crc["extract"].exec()

    # --- Bob: BSC LLR (the known-zero amplification happens in the decoder) ---
    sp_llr = np.zeros((1, N), dtype=np.float32)
    llr_bsc["llr_bsc ::input"].bind(padded_bob)
    llr_bsc["llr_bsc ::output"].bind(sp_llr)
    llr_bsc["llr_bsc"].exec()

    # --- Appending reconciliation loop ---
    leaked = CRC_SIZE
    failed = True
    stage_log = []
    frozen_acc = np.zeros((1, N), dtype=np.int32)
    corrected_key = None

    # Initialize the shortened positions in frozen_acc with their encoded
    # values (= 0 by the BGL construction, a public constant). The punctured
    # positions are NOT frozen in the decoder and nothing is ever
    # transmitted for them.
    frozen_acc[0, shortened_idx] = encoded_alice[0, shortened_idx]  # = 0

    print(f"\n  Initial leaked (CRC): {leaked}, "
          f"eff_assumed = {leaked/assumed_ideal_frozen:.6f}")

    for stage in range(len(frozen_vecs)):
        mask = eff_frozen_masks[stage]
        if stage == 0:
            new_idx = np.where(mask == 1)
        else:
            prev = eff_frozen_masks[stage - 1]
            new_idx = np.where(np.logical_xor(mask, prev).astype(np.int32) == 1)
        n_new = int(len(new_idx[1]))

        vals = encoded_alice[new_idx].reshape(1, -1).astype(np.int32)
        leaked += n_new
        frozen_acc[new_idx] = vals

        print(f"\n  --- Stage {stage+1}/{len(frozen_vecs)}: "
              f"f_cfg = {EFFICIENCY_STAGE[stage]:.2f} ---")
        print(f"      New frozen bits: {n_new}, cumulative leaked: {leaked}")
        print(f"      eff_assumed = {leaked/assumed_ideal_frozen:.6f}, "
              f"eff_real = {leaked/ideal_frozen_real:.6f}")

        crc_dec = CRC_polynomial(N, '0x04C11DB7', CRC_SIZE)
        dec = Decoder_polar_SCL_naive_CA(
            int(dec_K[stage]), N, SCL_LEN,
            frozen_vecs[stage], crc_dec,
        )
        dec.set_frozen_bits(frozen_vecs[stage])
        dec["set_known_bits ::KNOWN_BITS"].bind(known_bits_mask)
        dec["set_known_bits"].exec()
        dec["set_crc_const ::V_crc"].bind(crc_tag)
        dec["set_crc_const"].exec()

        dec_status  = np.ones((1, 1), dtype=np.int32)
        decoded_bob = np.zeros((1, N), dtype=np.int32)
        dec["decode_siho_cw_flexible_frozen ::F_N"].bind(frozen_acc)
        dec["decode_siho_cw_flexible_frozen ::Y_N"].bind(sp_llr)
        dec["decode_siho_cw_flexible_frozen ::status"].bind(dec_status)
        dec["decode_siho_cw_flexible_frozen ::V_N"].bind(decoded_bob)

        t0 = time.time()
        dec["decode_siho_cw_flexible_frozen"].exec()
        dt = time.time() - t0

        # Recover the U-side estimate: the polar generator is involutory over
        # GF(2), so re-encoding the decoded codeword yields the decoded U. The
        # corrected key = U at effective (non-sp) positions -- saved regardless
        # of CRC success (only when --save-result is given).
        if SAVE_RESULT:
            decoded_u = np.zeros_like(decoded_bob)
            enc["light_encode ::U_N"].bind(decoded_bob)
            enc["light_encode ::X_N"].bind(decoded_u)
            enc["light_encode"].exec()
            corrected_key = decoded_u.flatten()[non_sp_1d].copy()

        ok = bool(np.abs(dec_status).sum() == 0)
        ber_blk = float(np.logical_xor(decoded_bob, encoded_alice).sum() / N)
        ber_key = float(
            np.logical_xor(
                decoded_bob.flatten()[non_sp_1d],
                encoded_alice.flatten()[non_sp_1d],
            ).sum() / L_DATA
        )
        print(f"      Decode: {dt:.1f}s, status={int(dec_status.flatten()[0])} "
              f"({'PASS' if ok else 'FAIL'}), BER_key = {ber_key:.4e}")

        stage_log.append({
            'stage': stage + 1,
            'f_cfg': float(EFFICIENCY_STAGE[stage]),
            'frozen_eff': int(eff_frozen[stage]),
            'K': int(info_bits[stage]),
            'leaked': int(leaked),
            'eff_assumed': float(leaked / assumed_ideal_frozen),
            'eff_real': float(leaked / ideal_frozen_real),
            't': float(dt),
            'pass': ok,
            'ber_key': ber_key,
            'ber_block': ber_blk,
        })

        if ok:
            failed = False
            print(f"      *** SUCCESS at f_cfg = {EFFICIENCY_STAGE[stage]:.2f} ***")

        del dec, crc_dec, decoded_bob
        gc.collect()

    del enc, crc_alice, llr_bsc, encoded_alice, crc_tag, sp_llr
    del frozen_acc, frozen_vecs, eff_frozen_masks
    gc.collect()

    return {'stages': stage_log, 'failed': failed, 'assumed_q_ber': assumed_q_ber,
            'corrected_key': corrected_key}


# ============================================================
# Run pipeline
# ============================================================
print("\n" + "=" * 70)
print(f"RUNNING AIR PIPELINE FOR ZZ KEYGEN DATA")
print(f"  Assumed BER: {ASSUMED_BER}, S={S_BITS}, P={P_BITS}, SCL={SCL_LEN}")
print("=" * 70)

result = run_air_pipeline(ASSUMED_BER)

# ============================================================
# Final summary
# ============================================================
print("\n\n" + "#" * 70)
print("FINAL SUMMARY")
print("#" * 70)
print(f"Real BER (keygen, after 3-step flip): {ir_ber:.6f}")
print(f"N = {N}, L = {L_DATA}, S = {S_BITS}, P = {P_BITS}")
print(f"SCL = {SCL_LEN}, CRC = {CRC_SIZE}")
print(f"Shannon limit (real BER): {ideal_frozen_real:.1f} bits")
print()

if result is None:
    print("  [pipeline did not run — .pc file missing]")
else:
    print(f"--- Assumed BER = {result['assumed_q_ber']} ---")
    print(f"  {'Stg':>3} | {'f_cfg':>5} | {'F_eff':>8} | {'K':>8} | "
          f"{'leaked':>8} | {'eff_a':>8} | {'eff_r':>8} | "
          f"{'CRC':>4} | {'BER_key':>9} | {'t(s)':>6}")
    for s in result['stages']:
        print(f"  {s['stage']:>3} | {s['f_cfg']:>5.2f} | {s['frozen_eff']:>8} | "
              f"{s['K']:>8} | {s['leaked']:>8} | {s['eff_assumed']:>8.4f} | "
              f"{s['eff_real']:>8.4f} | "
              f"{'PASS' if s['pass'] else 'FAIL':>4} | {s['ber_key']:>9.2e} | "
              f"{s['t']:>6.1f}")
    if result['failed']:
        print(f"  => ALL STAGES FAILED (max f = {EFFICIENCY_STAGE[-1]:.2f})")
    else:
        succ = next(s for s in result['stages'] if s['pass'])
        print(f"  => Threshold at f_cfg = {succ['f_cfg']:.2f} "
              f"(eff_assumed = {succ['eff_assumed']:.4f}, "
              f"eff_real = {succ['eff_real']:.4f})")

    if SAVE_RESULT:
        base = os.path.splitext(log_file)[0]
        alice_out = base + '_alice.npy'
        bob_out = base + '_bob_corrected.npy'
        np.save(alice_out, alice_result_zz.astype(np.int32))
        print(f"\n  Alice sifted key saved      : {alice_out}")
        if result['corrected_key'] is not None:
            np.save(bob_out, result['corrected_key'])
            print(f"  Bob corrected key saved     : {bob_out}")
        else:
            print(f"  [no decode output — Bob corrected key not saved]")

print(f"\nTotal elapsed: {datetime.now() - start_time}")
print(f"Log file:      {log_file}")
