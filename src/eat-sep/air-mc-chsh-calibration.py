#!/usr/bin/env python3
r"""
AIR Monte-Carlo simulation for **CHSH rounds** in the separate-IR pipeline.

Calibration-driven MC pipeline:
  1. Analyse lab CHSH calibration count matrices
     -> noise BER (point estimate)
  2. Union bound on polar-code frozen bits (pure-shortening reliability files)
     -> 3 AIR efficiency stages targeting 75%, 25%, 0.001% success
  3. Pipeline-based MC simulation (py_aff3ct multi-threaded)

Only CHSH calibration data is used (no Gamma weighting, no ZZ counts) because
each IR codec in the separate pipeline is designed independently.

Pure shortening (BGL construction):
  - Mother code N = 2^15 = 32768
  - L = 19897 (expected CHSH data length, derived from experimental data shape)
  - S = N - L = 12871 shortened bits, P = 0 punctured bits
  - PE/PC files come from polar-fbgen-rust's batch method (--num-shortened)

Memory profile (N=2^15, SCL=32):
  - Each Decoder_polar_SCL_naive_CA: ~0.5 GB
  - Encoder + CRC + monitors + custom modules: ~0.3 GB total
  - 3-stage pipeline with SCL=32: ~2 GB peak (PER PIPELINE REPLICA)
  - Compare to N=2^20 case: ~9 GB per decoder, ~27 GB for 3 stages at SCL=4
  - NOTE: the py_aff3ct Pipeline replicates the whole chain once per
    --n-threads worker, so the process RSS scales ~linearly with
    --n-threads (measured ~1.6 GB/thread at SCL=32).
    Total budget = n_threads x per-replica memory.

Usage:
  python air-mc-chsh-calibration.py
  python air-mc-chsh-calibration.py --num-samples 128 --n-threads 8
  python air-mc-chsh-calibration.py --scl-len 64
  python air-mc-chsh-calibration.py --eff-stages 1.04 1.08
"""
import argparse
import math
import os
import sys
import time
from copy import deepcopy
from datetime import datetime

sys.path.insert(0, '../../submodules/py_aff3ct/build/lib')
import numpy as np
import py_aff3ct

sys.path.insert(0, '../../src/')
from utils import (
    num_frozen_bits_from_union_bound_from_file,
    union_failure_bound_from_file,
    generate_shortened_from_file, generate_punctured_from_file,
    generate_shortened_BGL, generate_punctured_BGL,
    generate_frozen_position_from_file,
    llr_calculater_bsc,
    BSC_digital_p, extractor, signal_waiter2, mask_fillor, mask_selector,
    control_identity, control_flusher, seq_execution,
)
from py_aff3ct.module.encoder import Encoder_polar
from py_aff3ct.module.decoder import Decoder_polar_SCL_naive_CA
from py_aff3ct.module.crc import CRC_polynomial
from py_aff3ct.module.monitor import Monitor_BFER_AR

# ============================================================
# Hardcoded CHSH calibration data (Sep — corrected laser-unlock artifact)
# 4 standard Bell-test subgroups; total 5997 rounds (4 main subgroups of 6086)
# ============================================================
CHSH_COUNTS = np.array([
    [156, 733, 798, 162],
    [766, 187, 144, 836],
    [471,  92,  87, 455],
    [440, 118,  91, 461],
])


def calibrate_chsh():
    """Return the noise BER (QBER_E point estimate) from CHSH calibration counts.

    QBER_E = sum of per-subgroup disagreement after optimal flip, divided by
    the total number of CHSH calibration rounds.
    """
    n_E = CHSH_COUNTS.sum()
    err_E = 0
    for cells in CHSH_COUNTS:
        n00, n01, n10, n11 = cells
        agree, disagree = n00 + n11, n01 + n10
        E = (agree - disagree) / cells.sum()
        err_E += int(disagree if E > 0 else agree)
    return float(err_E / n_E)


def union_bound_efficiencies_chsh(N, L, S, P, assumed_ber, pe_dir,
                                  last_eps=1e-5):
    """Compute 3 efficiency stages from union bound on joint S+P polar code.

    Reads the frozen_bits TSV file produced by polar-fbgen-rust's batch method
    (conditioned on the joint S+P pattern), then computes the minimal K_frozen
    to achieve target success rates of 75%, 25%, and (1-last_eps).
    """
    n_pow = int(math.log2(N))
    pe_file = os.path.join(
        pe_dir,
        f'frozen_bits_mu=100_N={n_pow}_p={assumed_ber}'
        f'_shortened={S}_punctured={P}.txt',
    )
    if not os.path.exists(pe_file):
        raise FileNotFoundError(
            f"PE file not found: {pe_file}\n"
            f"Generate it via polar-fbgen-rust batch method with "
            f"--num-shortened {S} --num-punctured {P} -p {assumed_ber} -N {n_pow}"
        )

    # Load sp positions (union for union-bound masking)
    sp_file_S = os.path.join(pe_dir, f'shortened_positions_N={n_pow}_S={S}.txt')
    sp_file_P = os.path.join(pe_dir, f'punctured_positions_N={n_pow}_P={P}.txt')
    if os.path.exists(sp_file_S):
        short_idx = generate_shortened_from_file(sp_file_S)
    else:
        short_idx = generate_shortened_BGL(N, S)
    if os.path.exists(sp_file_P):
        punct_idx = generate_punctured_from_file(sp_file_P)
    else:
        punct_idx = generate_punctured_BGL(N, P)
    sp_idx = np.concatenate([short_idx, punct_idx])

    H2 = lambda p: -p * math.log2(p) - (1 - p) * math.log2(1 - p) if 0 < p < 1 else 0
    shannon = L * H2(assumed_ber)

    stages = []
    for eps in [0.75, 0.25, last_eps]:
        Kf = num_frozen_bits_from_union_bound_from_file(pe_file, sp_idx, eps)
        Ki = L - Kf
        stages.append({'K_frozen': Kf, 'K_info': Ki,
                       'expected_FER': float(eps),
                       'efficiency': float(Kf / shannon)})
    return stages, shannon


def union_bound_failure_rates_chsh(N, L, S, P, assumed_ber, pe_dir,
                                   frozen_vector_list, sp_idx, num_frozen_bits_list):
    """Compute union-bound FER for each given stage configuration."""
    assert len(frozen_vector_list) == len(num_frozen_bits_list), \
        "Frozen vector list size mismatch with num_frozen_bits list"

    n_pow = int(math.log2(N))
    pe_file = os.path.join(
        pe_dir,
        f'frozen_bits_mu=100_N={n_pow}_p={assumed_ber}'
        f'_shortened={S}_punctured={P}.txt',
    )
    failure_rates = []
    for stage in range(len(num_frozen_bits_list)):
        failure_rates.append(union_failure_bound_from_file(
            pe_file, frozen_vector_list[stage],
            sp_idx, num_frozen_bits_list[stage]))
    return failure_rates


# ============================================================
# CLI
# ============================================================
parser = argparse.ArgumentParser(
    description='Calibration-driven AIR MC simulation for CHSH rounds (Sep).')
parser.add_argument('--num-samples', type=int, default=10000,
                    help='Number of MC frames (default: 10000).')
parser.add_argument('--scl-len', type=int, default=32,
                    help='SCL decoder list size (default: 32).')
parser.add_argument('--crc-size', type=int, default=32,
                    help='CRC tag length in bits (default: 32).')
parser.add_argument('--n-threads', type=int, default=20,
                    help='Pipeline compute-stage threads (collect fixed at 1)')
parser.add_argument('--pc-dir', type=str, default='../../conf/polar/sp-bsc/',
                    help='Directory of .pc and frozen_bits files.')
parser.add_argument('--pe-dir', type=str, default='../../conf/polar/sp-bsc/',
                    help='Directory of frozen_bits TSV files (for union bound).')
parser.add_argument('--log-dir', type=str, default='../../log/eat-diqkd-sep/',
                    help='Directory for log files.')
parser.add_argument('--seed', type=int, default=None,
                    help='Random seed for reproducibility.')
parser.add_argument('--eff-stages', type=float, nargs='+', default=None,
                    help='Efficiency stages except the last (default: None, ' +
                         'decided by union bound targeting 75% / 25% success).')
parser.add_argument('--failure-rate', type=float, default=1e-5,
                    help='Target failure rate of AIR (last stage). ' +
                         'Default 1e-5 -> 99.999% success.')
parser.add_argument('--assumed-ber', type=float, default=0.1825,
                    help='Assumed BER (default: 0.1825, as in the demo notebooks).')
parser.add_argument('--noise-ber', type=float, default=None,
                    help='Noise BER for MC simulation (default: CHSH point estimate).')
parser.add_argument('--S', type=int, default=12871,
                    help='Number of shortened bits (default: 12871, pure shortening).')
parser.add_argument('--P', type=int, default=0,
                    help='Number of punctured bits (default: 0).')

args = parser.parse_args()
NUM_SAMPLES = args.num_samples
SCL_LEN = args.scl_len
CRC_SIZE = args.crc_size
N_THREADS = [args.n_threads, 1]
PC_DIR  = args.pc_dir
PE_DIR  = args.pe_dir
LOG_DIR = args.log_dir
EFF_STAGES_ARG = args.eff_stages
LAST_EPS = args.failure_rate
S_BITS = args.S
P_BITS = args.P

if args.seed is not None:
    np.random.seed(args.seed)

# ============================================================
# Step 1 + 2: Calibration -> parameters
# ============================================================
CALIB_NOISE_BER = calibrate_chsh()
ASSUMED_BER = args.assumed_ber
NOISE_BER   = args.noise_ber if args.noise_ber is not None else CALIB_NOISE_BER

# Fixed code parameters for CHSH separate-IR codec
N = 1 << 15                       # 32768 (mother code length)
L = 19897                         # expected CHSH data length
assert S_BITS + P_BITS == N - L, \
    f"S+P must equal N-L: {S_BITS}+{P_BITS} != {N}-{L}"

n_pow = 15
shannon_H2 = lambda x: -x * np.log2(x) - (1 - x) * np.log2(1 - x) if 0 < x < 1 else 0.0

# Union-bound-derived efficiency stages
ub_stages, shannon_limit = union_bound_efficiencies_chsh(
    N, L, S_BITS, P_BITS, ASSUMED_BER, PE_DIR, last_eps=LAST_EPS)
EFF_STAGES = np.array([s['efficiency'] for s in ub_stages], dtype=np.float64)
FAILURE_RATE_BOUNDS = ['75%', '25%', str(LAST_EPS)]

if EFF_STAGES_ARG is not None:
    EFF_STAGES = np.array(EFF_STAGES_ARG + [ub_stages[-1]['efficiency']],
                          dtype=np.float64)

num_stages = len(EFF_STAGES)

# ============================================================
# Logger (early bind so all subsequent prints go to log)
# ============================================================
os.makedirs(LOG_DIR, exist_ok=True)
start_time = datetime.now()
_seed_tag = f'_seed{args.seed}' if args.seed is not None else ''
log_file = os.path.join(
    LOG_DIR,
    f'air_mc_chsh_ber{ASSUMED_BER:.4f}_noise{NOISE_BER:.6f}_'
    f'S{S_BITS}_P{P_BITS}_L{SCL_LEN}_n{NUM_SAMPLES}{_seed_tag}_'
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
print(f"Log file: {log_file}")
print(f"Started at: {start_time}")
print(f"\n=== CHSH AIR MC (Sep, separate IR) ===")
print(f"  Calibrated noise BER (point estimate): {CALIB_NOISE_BER:.6f}")
print(f"  Active assumed BER  : {ASSUMED_BER}")
print(f"  Active noise BER    : {NOISE_BER:.6f}")
print(f"  N / L / S / P       : {N} / {L} / {S_BITS} / {P_BITS}")
print(f"  Eff stages          : {[f'{e:.4f}' for e in EFF_STAGES]}")
print(f"  Num samples         : {NUM_SAMPLES}")
print(f"  SCL L / CRC         : {SCL_LEN} / {CRC_SIZE}")
print(f"  n_threads           : {N_THREADS}")
print(f"  RNG seed            : {args.seed}")

# ============================================================
# Load sp positions + masks
# ============================================================
sp_file_S = os.path.join(PE_DIR, f'shortened_positions_N={n_pow}_S={S_BITS}.txt')
sp_file_P = os.path.join(PE_DIR, f'punctured_positions_N={n_pow}_P={P_BITS}.txt')
if os.path.exists(sp_file_S):
    shortened_idx = generate_shortened_from_file(sp_file_S)
else:
    shortened_idx = generate_shortened_BGL(N, S_BITS)
if os.path.exists(sp_file_P):
    punctured_idx = generate_punctured_from_file(sp_file_P)
else:
    punctured_idx = generate_punctured_BGL(N, P_BITS)

shortened_vec = np.zeros(N, dtype=bool)
shortened_vec[shortened_idx] = True
punctured_vec = np.zeros(N, dtype=bool)
punctured_vec[punctured_idx] = True
sp_vec = shortened_vec | punctured_vec
sp_idx = np.concatenate([shortened_idx, punctured_idx])

# Mask for padder: 1 at non-sp (transmitted) positions
non_sp_mask = (~sp_vec).astype(np.int32).reshape((1, -1))

# (The frozen selector mask is built per stage in the loop below: eff_frozen only.)

# ============================================================
# Load .pc file + frozen vectors per stage
# ============================================================
fb_file = os.path.join(
    PC_DIR,
    f'n{n_pow}_sp-bsc_ber{ASSUMED_BER}_mu100_shortened{S_BITS}_punctured{P_BITS}.pc',
)
if not os.path.exists(fb_file):
    raise FileNotFoundError(
        f"PC file not found: {fb_file}\n"
        f"Generate it via: python data/pe2idx.py --input <frozen_bits.txt> "
        f"--output {fb_file} --channel-type sp-bsc --ber {ASSUMED_BER}"
    )

print(f"\nLoading polar code: {fb_file}")
with open(fb_file, 'r') as f:
    reliability_sorted_idx = np.array(
        [int(x) for x in f.readlines()[3].split()], dtype=np.int64)
assert reliability_sorted_idx.shape[0] == N

assumed_ideal = L * shannon_H2(ASSUMED_BER)
real_ideal = L * shannon_H2(NOISE_BER)
crc_gain = CRC_SIZE / assumed_ideal

eff_stage_cfg = EFF_STAGES - crc_gain
eff_frozen = np.asarray(eff_stage_cfg * assumed_ideal, dtype=np.int64)
info_bits = N - eff_frozen - S_BITS - P_BITS
# Decoder information positions: AIR information (P = 0: no free punctured leaves)
dec_K = N - eff_frozen - S_BITS
assert (info_bits > 0).all(), f"info_bits must be > 0, got {info_bits}"
assert np.all(np.diff(eff_frozen) > 0), "eff_frozen must be strictly increasing"

print(f"  H2(assumed) = {shannon_H2(ASSUMED_BER):.6f}, L*H2 = {assumed_ideal:.1f}")
print(f"  H2(noise)   = {shannon_H2(NOISE_BER):.6f}, L*H2 = {real_ideal:.1f}")
print(f"  CRC gain    : {crc_gain:.6f}")
print(f"  Eff frozen  : {eff_frozen.tolist()}")
print(f"  Info bits   : {info_bits.tolist()}")

# Build frozen vectors per stage:
#   frozen_vec[stage] = shortened_vec | eff_frozen_vec[stage]
#     (the punctured positions are NOT decoder-frozen: their codeword bits
#      are key-dependent and never transmitted, their leaves are decoded
#      freely, guided by the amplified known-zero LLR on the input side)
#   eff_frozen_mask[stage] = eff_frozen positions only (length-N int32)
#   frozen_select_mask[stage] = eff_frozen only (for selector/fillor)
frozen_vec_stage, eff_frozen_vec_stage = [], []
eff_frozen_mask, frozen_select_mask = [], []
for stage in range(num_stages):
    _, _, eff_idx = generate_frozen_position_from_file(
        fb_file, int(eff_frozen[stage]), sp_idx, verbose=False)

    fv = shortened_vec.copy()
    fv[eff_idx] = True
    ev = np.zeros(N, dtype=bool)
    ev[eff_idx] = True

    frozen_vec_stage.append(fv)
    eff_frozen_vec_stage.append(ev)
    eff_frozen_mask.append(deepcopy(ev.astype(np.int32).reshape((1, -1))))

    # Mask for frozen_selector/frozen_fillor: the eff_frozen syndrome bits.
    # Number of selected positions = eff_frozen[stage].
    frozen_select_mask.append(deepcopy(ev.astype(np.int32).reshape((1, -1))))

if EFF_STAGES_ARG is not None:
    FAILURE_RATE_BOUNDS = union_bound_failure_rates_chsh(
        N, L, S_BITS, P_BITS, ASSUMED_BER, PE_DIR,
        frozen_vec_stage, sp_idx, eff_frozen)
print(f"  Failure-rate bounds : {FAILURE_RATE_BOUNDS}")

real_leaked = eff_frozen + CRC_SIZE
real_eff_stages = np.array([
    (int(eff_frozen[s]) + CRC_SIZE) / real_ideal for s in range(num_stages)
])
print(f"  Real leaked bits    : {real_leaked.tolist()}")
print(f"  Real eff stages     : {real_eff_stages.tolist()}")

# ============================================================
# Build modules (all pre-built; freed only at program exit)
# ============================================================
print(f"\nBuilding {num_stages}-stage pipeline modules (SCL={SCL_LEN}) ...")

src_alice = py_aff3ct.module.source.Source_random_fast(L)
src_bob = BSC_digital_p(NOISE_BER, L)
padder_alice = mask_fillor(L, N, dtype=np.int32)
padder_bob = mask_fillor(L, N, dtype=np.int32)

enc_stage, crc_stage, dec_stage = [], [], []
ext_crc_stage, llr_bsc_stage = [], []
known_bits_mask_mc = sp_vec.astype(np.int32).reshape((1, -1))
waiter_stage = []
frozen_selector_stage, frozen_fillor_stage = [], []
mnt_stage, mnt_cascl_stage = [], []
trunc_alice_stage, trunc_llr_stage, trunc_mnt_stage = [], [], []

for stage in range(num_stages):
    enc = Encoder_polar(int(dec_K[stage]), N, frozen_vec_stage[stage])
    enc.set_frozen_bits(frozen_vec_stage[stage])
    enc_stage.append(enc)
    crc_stage.append(CRC_polynomial(N, '0x04C11DB7', CRC_SIZE))
    dec = Decoder_polar_SCL_naive_CA(
        int(dec_K[stage]), N, SCL_LEN,
        frozen_vec_stage[stage], crc_stage[stage])
    dec.set_frozen_bits(frozen_vec_stage[stage])
    dec_stage.append(dec)

    ext_crc_stage.append(extractor(N + CRC_SIZE, N, N + CRC_SIZE))
    # LLR calculator uses NOISE_BER (the actual channel noise rate for MC)
    llr_bsc_stage.append(llr_calculater_bsc(NOISE_BER, N))

    # The known-zero padding amplification happens inside the decoder
    # (set_known_bits, x17 like the former external shortener).
    dec_stage[stage]["set_known_bits ::KNOWN_BITS"].bind(known_bits_mask_mc)
    dec_stage[stage]["set_known_bits"].exec()

    waiter_stage.append(signal_waiter2(N, 1, in1_dtype=np.float32, in2_dtype=np.int32))
    mnt_stage.append(Monitor_BFER_AR(N, NUM_SAMPLES, max_n_frames=NUM_SAMPLES))
    mnt_cascl_stage.append(Monitor_BFER_AR(1, NUM_SAMPLES, max_n_frames=NUM_SAMPLES))

    n_select = int(eff_frozen[stage])
    frozen_selector_stage.append(mask_selector(N, n_select, dtype=np.int32))
    frozen_fillor_stage.append(mask_fillor(n_select, N, dtype=np.int32))

    trunc_alice_stage.append(control_identity(N, dtype=np.int32, control_dtype=np.int32))
    trunc_llr_stage.append(control_flusher(N, dtype=np.float32, control_dtype=np.int32))
    trunc_mnt_stage.append(control_identity(N, dtype=np.int32, control_dtype=np.int32))

# ============================================================
# Socket bindings
# ============================================================
print("Binding sockets ...")

src_bob["digital_bit_flip ::input_bits"].bind(src_alice["generate ::U_K"])
padder_alice["fill ::mask"].bind(non_sp_mask)
padder_alice["fill ::input"].bind(src_alice["generate ::U_K"])
padder_bob["fill ::mask"].bind(non_sp_mask)
padder_bob["fill ::input"].bind(src_bob["digital_bit_flip ::output_bits"])

success_status = np.zeros(1, dtype=np.int32).reshape(1, -1)

for stage in range(num_stages):
    # --- Encoder path: only run if previous stage failed ---
    if stage == 0:
        enc_stage[stage]["light_encode ::U_N"].bind(padder_alice["fill ::output"])
    else:
        trunc_alice_stage[stage]["control_identity ::input"].bind(
            padder_alice["fill ::output"])
        trunc_alice_stage[stage]["control_identity ::control"].bind(
            dec_stage[stage - 1]["decode_siho_cw_flexible_frozen ::status"])
        enc_stage[stage]["light_encode ::U_N"].bind(
            trunc_alice_stage[stage]["control_identity ::output"])

    # --- CRC path ---
    crc_stage[stage]["build ::U_K1"].bind(enc_stage[stage]["light_encode ::X_N"])
    ext_crc_stage[stage]["extract ::input"].bind(crc_stage[stage]["build ::U_K2"])
    dec_stage[stage]["set_crc_const ::V_crc"].bind(
        ext_crc_stage[stage]["extract ::output"])
    waiter_stage[stage]["wait ::input_2"].bind(
        dec_stage[stage]["set_crc_const ::status"])

    # --- Frozen value extraction (eff_frozen syndrome positions) ---
    frozen_selector_stage[stage]["select ::input"].bind(
        enc_stage[stage]["light_encode ::X_N"])
    frozen_selector_stage[stage]["select ::mask"].bind(frozen_select_mask[stage])
    frozen_fillor_stage[stage]["fill ::input"].bind(
        frozen_selector_stage[stage]["select ::output"])
    frozen_fillor_stage[stage]["fill ::mask"].bind(frozen_select_mask[stage])

    # --- Bob LLR path: BSC LLR -> waiter (padding amplification in the decoder) ---
    llr_bsc_stage[stage]["llr_bsc ::input"].bind(padder_bob["fill ::output"])
    waiter_stage[stage]["wait ::input_1"].bind(
        llr_bsc_stage[stage]["llr_bsc ::output"])

    # --- Decoder inputs ---
    dec_stage[stage]["decode_siho_cw_flexible_frozen ::F_N"].bind(
        frozen_fillor_stage[stage]["fill ::output"])

    if stage == 0:
        dec_stage[stage]["decode_siho_cw_flexible_frozen ::Y_N"].bind(
            waiter_stage[stage]["wait ::output"])
        mnt_stage[stage]["check_errors ::U"].bind(
            enc_stage[stage]["light_encode ::X_N"])
        mnt_stage[stage]["check_errors ::V"].bind(
            dec_stage[stage]["decode_siho_cw_flexible_frozen ::V_N"])
    else:
        # Truncate LLR feed: only decode if previous stage failed
        trunc_llr_stage[stage]["control_identity ::input"].bind(
            waiter_stage[stage]["wait ::output"])
        trunc_llr_stage[stage]["control_identity ::control"].bind(
            dec_stage[stage - 1]["decode_siho_cw_flexible_frozen ::status"])
        dec_stage[stage]["decode_siho_cw_flexible_frozen ::Y_N"].bind(
            trunc_llr_stage[stage]["control_identity ::output"])
        # Truncate monitor reference: always compare against stage-0 encoded
        trunc_mnt_stage[stage]["control_identity ::input"].bind(
            enc_stage[0]["light_encode ::X_N"])
        trunc_mnt_stage[stage]["control_identity ::control"].bind(
            dec_stage[stage - 1]["decode_siho_cw_flexible_frozen ::status"])
        mnt_stage[stage]["check_errors ::U"].bind(
            trunc_mnt_stage[stage]["control_identity ::output"])
        mnt_stage[stage]["check_errors ::V"].bind(
            dec_stage[stage]["decode_siho_cw_flexible_frozen ::V_N"])

    mnt_cascl_stage[stage]["check_errors ::U"].bind(success_status)
    mnt_cascl_stage[stage]["check_errors ::V"].bind(
        dec_stage[stage]["decode_siho_cw_flexible_frozen ::status"])

# ============================================================
# Pipeline
# ============================================================
initial_tasks = [src_alice["generate"]]
final_tasks = []
stages = [
    [initial_tasks, [],
     [m["check_errors"] for m in mnt_stage] + [m["check_errors"] for m in mnt_cascl_stage]],
    [[m["check_errors"] for m in mnt_stage] + [m["check_errors"] for m in mnt_cascl_stage],
     [], []]
]

seq = py_aff3ct.tools.pipeline.Pipeline(
    initial_tasks, final_tasks, stages,
    N_THREADS, [NUM_SAMPLES], [False],
)

for lt in seq.get_tasks_per_types():
    for t in lt:
        t.stats = True

for stage in range(num_stages):
    mnt_stage[stage].reset()
    mnt_cascl_stage[stage].reset()

# ============================================================
# Execute
# ============================================================
print(f"\nStarting MC simulation ({NUM_SAMPLES} frames, n_threads={N_THREADS}) ...\n")

try:
    se = seq_execution(seq=seq)
    se.start()

    print(f"  {'Frames':>8} | {'FE(st0)':>8} | {'FE(stL)':>8} | "
          f"{'CASCL(stL)':>10} | {'Elapsed':>8} | {'Remaining':>10}")
    print("  " + "-" * 70)

    while se.is_alive():
        time.sleep(2)
        elapsed = time.time() - se.starting_time
        total_fra = mnt_stage[-1].get_n_analyzed_fra()
        fe_0 = mnt_stage[0].get_n_fe()
        fe_l = mnt_stage[-1].get_n_fe()
        fe_cascl = mnt_cascl_stage[-1].get_n_fe()
        if total_fra > 20:
            remaining = max(float(NUM_SAMPLES - total_fra) / (total_fra / elapsed), 0)
        elif total_fra > 0:
            remaining = -1
        else:
            remaining = -1
        eta_str = (time.strftime('%H:%M:%S', time.gmtime(remaining))
                   if remaining >= 0 else '     warmup')
        print(f"  {total_fra:>8} | {fe_0:>8} | {fe_l:>8} | {fe_cascl:>10} | "
              f"{time.strftime('%H:%M:%S', time.gmtime(elapsed)):>8} | "
              f"{eta_str:>10}", end="\r")

    # ============================================================
    # Results
    # ============================================================
    print("\n\n" + "=" * 80)
    print("MC RESULTS")
    print("=" * 80)

    H2_assumed = shannon_H2(ASSUMED_BER)
    H2_noise   = shannon_H2(NOISE_BER)
    ideal_assumed = L * H2_assumed
    ideal_real    = L * H2_noise

    n_fe_vals = [NUM_SAMPLES]
    for stage in range(num_stages):
        n_fe_vals.append(mnt_stage[stage].get_n_fe())

    print(f"\n  Assumed BER: {ASSUMED_BER:.4f}  (H2={H2_assumed:.6f}, L*H2={ideal_assumed:.0f})")
    print(f"  Noise BER  : {NOISE_BER:.6f}  (H2={H2_noise:.6f}, L*H2={ideal_real:.0f})")
    print(f"  CRC size   : {CRC_SIZE} bits")
    print(f"  Code       : N={N}, L={L}, S={S_BITS}, P={P_BITS}")

    for stage in range(len(FAILURE_RATE_BOUNDS)):
        meta = ub_stages[stage]
        n_fe_frame = mnt_stage[stage].get_n_fe()
        n_fe_cascl = mnt_cascl_stage[stage].get_n_fe()
        n_pass = NUM_SAMPLES - n_fe_frame
        leaked = int(eff_frozen[stage]) + CRC_SIZE
        eff_target = leaked / ideal_assumed
        eff_real   = leaked / ideal_real

        print(f"\n  --- Stage {stage} ({FAILURE_RATE_BOUNDS[stage]} target, "
              f"f_cfg={EFF_STAGES[stage]:.4f}) ---")
        print(f"    Union-bound expected FER    : {meta['expected_FER']:.2e}")
        print(f"    K_frozen (effective)        : {int(eff_frozen[stage])}")
        print(f"    K_info                      : {int(info_bits[stage])}")
        print(f"    Leaked bits (K_frozen+CRC)  : {leaked}")
        print(f"    Efficiency (target, assumed): {eff_target:.6f}")
        print(f"    Efficiency (real, noise)    : {eff_real:.6f}")
        print(f"    Frames analysed             : {mnt_stage[stage].get_n_analyzed_fra()}")
        print(f"    Frame errors (dec!=enc)     : {n_fe_frame}")
        print(f"    Frames corrected            : {n_pass}")
        print(f"    FER (n_fe / N)               : {n_fe_frame/NUM_SAMPLES:.4e}")
        print(f"    CASCL failures (CRC mismatch): {n_fe_cascl}")

    # Average efficiency (assumed-BER weights)
    print(f"\n  === Average efficiency breakdown (assumed BER) ===")
    print(f"  avg_eff = sum_k f_k * (n_fe[k-1] - n_fe[k]) / N, N_init={NUM_SAMPLES}")
    avg_eff = 0.0
    for stage in range(num_stages):
        corrected = n_fe_vals[stage] - n_fe_vals[stage + 1]
        contrib = EFF_STAGES[stage] * corrected / NUM_SAMPLES
        avg_eff += contrib
        print(f"    Stage {stage}: f={EFF_STAGES[stage]:.4f}, "
              f"n_fe_prev={n_fe_vals[stage]}, n_fe={n_fe_vals[stage + 1]}, "
              f"corrected={corrected}, contrib={contrib:.6f}")
    final_fe = n_fe_vals[-1]
    final_contrib = EFF_STAGES[-1] * final_fe / NUM_SAMPLES
    avg_eff += final_contrib
    print(f"    Final  : f={EFF_STAGES[-1]:.4f}, n_fe[last]={final_fe}, "
          f"contrib={final_contrib:.6f}")
    print(f"\n  Average efficiency (assumed BER) : {avg_eff:.6f}")

    # Average efficiency (real-BER weights)
    avg_real_eff = 0.0
    for stage in range(num_stages):
        corrected = n_fe_vals[stage] - n_fe_vals[stage + 1]
        avg_real_eff += real_eff_stages[stage] * corrected / NUM_SAMPLES
    avg_real_eff += real_eff_stages[-1] * final_fe / NUM_SAMPLES
    print(f"  Average efficiency (noise BER)   : {avg_real_eff:.6f}")

    print(f"\n  Final FER (frame errors, stage {num_stages - 1}) : "
          f"{final_fe / NUM_SAMPLES:.4e}")
    final_cascl = mnt_cascl_stage[-1].get_n_fe()
    print(f"  Final FER (CASCL, stage {num_stages - 1})        : "
          f"{final_cascl / NUM_SAMPLES:.4e}")

    print(f"\nTotal elapsed: {datetime.now() - start_time}")
    print(f"Log file: {log_file}")

finally:
    # Cleanup — release py_aff3ct objects before Python finalization
    # (prevents Socket unbind crash on exit)
    import gc as _gc
    del se, seq
    enc_stage.clear(); dec_stage.clear(); crc_stage.clear()
    ext_crc_stage.clear(); llr_bsc_stage.clear()
    waiter_stage.clear()
    frozen_selector_stage.clear(); frozen_fillor_stage.clear()
    trunc_alice_stage.clear(); trunc_llr_stage.clear(); trunc_mnt_stage.clear()
    mnt_stage.clear(); mnt_cascl_stage.clear()
    del src_alice, src_bob, padder_alice, padder_bob
    _gc.collect()
