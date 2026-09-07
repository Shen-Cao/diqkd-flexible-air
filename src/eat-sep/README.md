# `src/eat-sep/` — Separate-IR DIQKD Information Reconciliation Programs

Programs for calibration analysis, real-data reconciliation, and Monte-Carlo
simulation of the **separate** Appending Information Reconciliation (AIR)
protocol for device-independent quantum key distribution.

In the separate-IR pipeline, CHSH testing rounds and ZZ key-gen rounds are
reconciled by **two independent polar codes**, each matched to its own BER regime:

| Codec | Data | N | L | BER regime | Default (S, P) | Default SCL |
|---|---|---|---|---|---|---|
| CHSH | ~19.9k Bell-test rounds | $2^{15} = 32768$ | $19897$ | $\sim 17\%$ | $(12871, 0)$ pure shortening | $32$ |
| ZZ keygen | ~861k Z⊗Z key-gen rounds | $2^{20} = 1048576$ | $861247$ | $\sim 3.4\%$ | $(187329, 0)$ pure shortening | $8$ |

Both codecs use BGL shortened polar codes (Bioglio, Gabry, Land 2017) for
rate compatibility: the trimmed positions are the bit-reversal **suffix** set
$\mathcal{S}$ (pure shortening), whose codeword bits are zero by construction
and therefore known to both parties. Channel reliability files live in
`conf/polar/sp-bsc/`.

---

## Programs

### 1. `data/data4eat-sep/calibration-data-analysis-sep.py` — Calibration Analysis & Assumed-QBER Estimator

**Purpose:** Estimate the **assumed QBER** of the two codecs from the lab
calibration data (lives next to the dataset in `data/data4eat-sep/`,
together with `diqkd-analysis-sep.py`):

- Analyse the calibration count matrices to derive the BER point estimates
  (QBER_E, Q) and their confidence intervals.
- Report the assumed QBER of each codec as CI upper bounds at
  90/95/99/99.9%, ceiled to 5e-4.

**Dependencies:** numpy only, calibration data hard-coded.

**Usage (from the repository root):**

```bash
python data/data4eat-sep/calibration-data-analysis-sep.py
python data/data4eat-sep/calibration-data-analysis-sep.py --gamma 0.02
```

**Output:**

- CHSH violation $S$ and $\sigma_S$
- $QBER_E$ (CHSH BER after per-subgroup optimal flip)
- $Q$ (key-gen BER)
- Legacy BER_combined (for single-codec comparison)
- Assumed-QBER tables: two-sided upper bounds at $90/95/99/99.9\%$, ceiled to $5 \times 10^{-4}$

---

### 2. `ir-raw-chsh.py` — CHSH Real-Data AIR Pipeline

**Purpose:** Run the AIR reconciliation pipeline on real CHSH testing data
from `data/data4eat-sep/`. Extracts CHSH rounds (~19.9k pairs), applies the
3-step bit-flip pipeline (SNSPD correction, per-subgroup CHSH flip, AIR flip),
then reconciles via a BGL **pure-shortening** polar code. The 12,871
shortened codeword bits are zero by construction, hence known to both parties
and never transmitted: the decoder amplifies the LLR at these known-zero
padding positions natively (its `set_known_bits` task) and hard-freezes the
corresponding codeword leaves.

**Dependencies:** py_aff3ct, `data/data4eat-sep/*.npy`,
`conf/polar/sp-bsc/*shortened*punctured*.pc` and the matching frozen_bits
PE table (union-bound final stage).

**Memory:** Each CA-SCL decoder ($N=2^{15}$) at SCL$=32$ uses ~0.5 GB. Peak
~1 GB. A stage decode takes a few seconds (no free punctured leaves).

**Usage:**

```bash
# Defaults: BER=0.1825, S=12871, P=0, SCL=32 (pure shortening, same as the demo notebooks)
python ir-raw-chsh.py

# Focus on specific efficiency stages
python ir-raw-chsh.py --eff-stages 1.04 1.05 1.06

# Save the reconciliation result as .npy
python ir-raw-chsh.py --save-result --eff-stages 1.05 1.10

# High-SCL experiment (SCL=64 or 128 also work)
python ir-raw-chsh.py 0.1825 0 12871 64

# Other BER / trimming combos need matching .pc files (see below)
python ir-raw-chsh.py 0.185 2868 10000 32
```

**Key CLI arguments:**

| Argument | Default | Description |
|---|---|---|
| `assumed_ber` (positional) | $0.1825$ | Assumed BER for polar code |
| `S` (positional) | $12871$ | Shortened bits |
| `P` (positional) | $0$ | Punctured bits |
| `scl_len` (positional) | $32$ | SCL decoder list size |
| `--eff-stages` | full list | Override the whole stage list |
| `--save-result` | off | Save `<log>_alice.npy` (Alice sifted key) and `<log>_bob_corrected.npy` (Bob corrected key, effective positions only; last attempted stage, regardless of CRC) |

The default stage schedule matches the demo notebooks: $f = 1.05$, $f = 1.10$,
and a final union-bound stage at failure rate $\varepsilon = 10^{-5}$.

The shipped default (BER=0.1825, pure shortening) is the configuration
verified end-to-end by the interactive demo (see section 7).

**Output:** Log in `log/eat-diqkd-sep/` with per-stage efficiency, BER_key,
CRC status, decode time.

---

### 3. `ir-raw-keygen.py` — ZZ Key-Gen Real-Data AIR Pipeline

**Purpose:** Run the AIR reconciliation pipeline on real ZZ key-gen data from
`data/data4eat-sep/`. Extracts key-gen rounds (~861k pairs), applies SNSPD
correction + AIR flip (CHSH flip skipped for Z-basis rounds), then reconciles
via a pure-shortening BGL polar code at $N=2^{20}$ ($S=187{,}329$, $P=0$; the code path still supports puncturing, but it is not used in the released configuration — see src/eat-sep/study-puncturing/README.md).

**Dependencies:** py_aff3ct, `data/data4eat-sep/*.npy`,
`conf/polar/sp-bsc/*shortened*punctured*.pc` and the matching frozen_bits
PE table (union-bound final stage; unpack it first, see below).

**Memory:** Each CA-SCL decoder ($N=2^{20}$, SCL$=8$) uses ~4.5 GB. Decoders are
freed via `del` + `gc.collect()` between stages $\to$ peak ~8 GB.

> [!CAUTION]
> Do NOT use SCL > 8 on machines with < 32 GB RAM.

**Usage:**

```bash
# Defaults: BER=0.0365, S=187329, P=0, SCL=8 (pure shortening, same as the demo notebooks)
python ir-raw-keygen.py

# Focus on specific efficiency stages
python ir-raw-keygen.py --eff-stages 1.04 1.05 1.06

# Save the reconciliation result as .npy
python ir-raw-keygen.py --save-result --eff-stages 1.05 1.10

# Other BERs need matching .pc files (see below)
python ir-raw-keygen.py 0.038 187329 0 8
```

**Key CLI arguments:**

| Argument | Default | Description |
|---|---|---|
| `assumed_ber` (positional) | $0.0365$ | Assumed BER for polar code |
| `S` (positional) | $187329$ | Shortened bits |
| `P` (positional) | $0$ | Punctured bits |
| `scl_len` (positional) | $8$ | SCL decoder list size |
| `--eff-stages` | full list | Override the whole stage list |
| `--save-result` | off | Save `<log>_alice.npy` (Alice sifted key) and `<log>_bob_corrected.npy` (Bob corrected key, effective positions only; last attempted stage, regardless of CRC) |

The default stage schedule matches the demo notebooks: $f = 1.05$, $f = 1.10$,
and a final union-bound stage at failure rate $\varepsilon = 10^{-5}$.

The shipped default (BER=0.0365, pure shortening) is the configuration verified
end-to-end by the interactive demo (see section 6).

**Output:** Log in `log/eat-diqkd-sep/` with per-stage efficiency, BER_key,
CRC status, decode time (expect ~5 min/stage at $N=2^{20}$, SCL$=8$).

---

### 4. `air-mc-chsh-calibration.py` — CHSH Monte-Carlo Simulation

**Purpose:** Estimate the average efficiency and failure rate of the CHSH AIR
scheme by simulating many frames of synthetic BSC data at the calibration-derived
noise BER. Uses py_aff3ct Pipeline for multi-threaded execution. All stage
decoders are pre-built and held in memory (freed only at exit).

**Role:** as the file name suggests, this validates the parameter setting
**before** the experimental data is taken, from calibration data alone: both
the synthetic channel and the decoder LLRs use the calibration noise-BER
point estimate, assumed to be close to the realized QBER of the experiment.
(The real-data programs and the demos compute their LLRs at the assumed QBER
instead.)

**Reproducibility:** the committed 10,000-frame run was produced with
`--seed 39`:
`python air-mc-chsh-calibration.py --num-samples 10000 --n-threads 100 --eff-stages 1.05 1.10 --seed 39`.
Both Monte-Carlo programs print the RNG seed into the log header.

**Dependencies:** py_aff3ct, `conf/polar/sp-bsc/*shortened*punctured*.pc` and
`.txt` files (for union bound).

**Memory:** ~0.5 GB per decoder at $N=2^{15}$, SCL$=32$; one 3-stage pipeline
replica is ~2 GB (measured ~1.6 GB). The py_aff3ct `Pipeline` replicates the
entire chain once per `--n-threads` worker, so the process RSS scales
~linearly with `--n-threads` (measured 1.65 GB at 1 thread, 6.4 GB at
4 threads): the committed 10,000-frame run at `--n-threads 100` needs ~160 GB
and finishes in ~55 minutes.

**Committed results** (`log/eat-diqkd-sep/mc_chsh_n10000/`, seed 39): stage-0
FER $5.53\times10^{-2}$ (the union-bound figure for $f=1.05$ is $0.75$, so the
bound is very loose), stage 1 and stage 2 correct the remainder,
**0/10,000 frames abort**, and the average efficiency is **1.0528**
(assumed-BER basis) / **1.0863** (noise-BER basis). The final union-bound
stage of the pure-shortening configuration needs $f = 1.1988$, i.e.
$16{,}319$ frozen bits.

> [!NOTE]
> **Sample independence.** The `Pipeline` clones the whole chain once per worker,
> and `Source_random_fast::clone()` copies the PRNG state, so before aff3ct
> `90acd1198` every worker replayed the same source stream and the Monte-Carlo
> effective sample size was `n_frames / n_threads`. With that fix each clone gets
> a distinct seed and the workers draw independent streams.

**Usage:**

```bash
# Defaults: assumed BER=0.1825, S=12871, P=0, SCL=32, 10000 frames, 20 threads
python air-mc-chsh-calibration.py

# Quick test
python air-mc-chsh-calibration.py --num-samples 128 --n-threads 4

# Custom stages and stronger decoder
python air-mc-chsh-calibration.py --eff-stages 1.05 1.10 --scl-len 64
```

**Key CLI arguments:**

| Argument | Default | Description |
|---|---|---|
| `--num-samples` | $10000$ | Number of MC frames |
| `--scl-len` | $32$ | SCL decoder list size |
| `--n-threads` | $20$ | Pipeline compute threads |
| `--assumed-ber` | $0.1825$ | Assumed BER (as in the demo notebooks) |
| `--noise-ber` | calib point estimate | Override MC noise BER |
| `--eff-stages` | union bound | Override efficiency stages |
| `--S` / `--P` | $12871 / 0$ | Shortened / punctured bits |

**Output:** Log in `log/eat-diqkd-sep/` with per-stage FER, CRC failures,
average efficiency (both assumed-BER and noise-BER weighted), and timing.

---

### 5. `air-mc-keygen-calibration.py` — ZZ Key-Gen Monte-Carlo Simulation

**Purpose:** Same role as `air-mc-chsh-calibration.py` but for the ZZ keygen
codec at $N=2^{20}$. Simulates synthetic BSC data at the ZZ-calibration-derived
noise BER (~3.3%).

**Reproducibility:** the committed runs use single seeds 40--49, one
1,000-frame run per seed (`log/eat-diqkd-sep/mc_keygen_n10000/`); the seed
appears in each log file name and is now also printed in the log header.

**Committed results** (each run 1,000 frames at `--n-threads 32`): stage-0 FER
$0.405$--$0.450$ (the design expects roughly $0.42$, matching the union-bound
figure for $f=1.05$), stage-1 FER $0.003$--$0.009$, **stage 2 corrects every
remaining frame (0/10,000 failures)**, and the average efficiency across the
ten seeds is **1.0726** $\pm$ $0.0006$ (assumed-BER basis) / **1.1556** $\pm$
$0.0006$ (noise-BER basis). The per-seed table is
`..._S187329_P0_L8_n1000_10.md`.

**Dependencies:** py_aff3ct, `conf/polar/sp-bsc/*N=20*.pc` and `.txt` files.

**Memory:** **~4.5 GB per decoder at $N=2^{20}$, SCL$=8$.** One 3-stage
pipeline replica is ~16 GB, and the py_aff3ct `Pipeline` replicates the entire
chain once per `--n-threads` worker, so the process RSS scales ~linearly with
`--n-threads`. **Budget RAM as `n_threads × ~16 GB`.**

Memory estimates by SCL:

| SCL | Per-decoder | 3-stage replica | × `n_threads` |
|---|---|---|---|
| 8 | 4.5 GB | 16 GB | × n_threads |
| 16 | 9 GB | 30 GB | × n_threads |
| 32 | 18 GB | 56 GB | × n_threads |
| 64 | 36 GB | 110 GB | × n_threads |

> [!CAUTION]
> Larger SCL list size will not only quickly increase the RAM occupation, but
> also significantly increase required time for decoding.
>
> All figures are **per pipeline replica**. Since the py_aff3ct `Pipeline`
> replicates the whole chain once per `--n-threads` worker, the total process
> RSS is approximately `n_threads ×` the values above. Choose `--n-threads`
> from the RAM budget first, then check the CPU budget.

**Usage (on remote server):**

```bash
# Defaults: assumed BER=0.0365, S=187329, P=0, SCL=8, 10000 frames
python air-mc-keygen-calibration.py --num-samples 10000 --n-threads 32

# Push decoder strength (needs high-RAM server)
python air-mc-keygen-calibration.py --scl-len 32 --n-threads 32

# Custom efficiency stages
python air-mc-keygen-calibration.py --eff-stages 1.04 1.06
```

**Key CLI arguments:** Same as `air-mc-chsh-calibration.py` but with defaults
suited for ZZ keygen (`--assumed-ber 0.0365 --S 187329 --P 0 --scl-len 8`).

**Output:** Same format as CHSH MC.

---

### 6. `qkd-air-demo-alice-keygen.ipynb` / `qkd-air-demo-bob-keygen.ipynb` — Interactive AIR Demo on Real Keygen Data

**Purpose:** Jupyter demonstration of the actual error-correction process over
classical communication for the ZZ key-gen codec. The two
notebooks play the two parties of the AIR protocol on the real keygen
experimental data (`data/data4eat-sep/`): Alice encodes her (AIR-flipped)
key and streams the CRC tag + appended frozen bits over HTTP; Bob applies the
SNSPD correction, computes BSC LLRs, decodes with CA-SCL and reports the CRC
status back. They share the same parameter cell and **must** be run with
matching parameters.

**Defaults (same convention as `air-mc-keygen-calibration.py`):**

| Parameter | Value | Note |
|---|---|---|
| Assumed QBER | $0.0365$ | $95\%$ Wald upper bound on the ZZ calibration BER (point $0.033115$, $\sigma=0.001684$, upper $0.036415$), ceiled to $5 \times 10^{-4}$ (same ceiling as the calibration program). Reliability file exists at exactly this BER. |
| Stages | $f = 1.05, f = 1.10$ | configured efficiencies vs. assumed ideal |
| Final stage | union bound @ $99.999\%$ | minimal frozen bits for failure rate $\varepsilon = 10^{-5}$ (the MC default `--failure-rate`) |
| SCL / CRC | $8 / 32$ | |
| Trimming | $S = 187329, P = 0$ | pure shortening |

**Usage:**

1. Start **Bob's** notebook first (listens on port 8003, sends on 8012).
2. Run **Alice's** notebook (listens on port 8012, sends on 8003).
3. Check the logs in `log/eat-diqkd-sep/ir_demo_keygen/`:
   `qkd_air_demo_{alice,bob}_keygen_assumed_0.0365_*.txt`.

**Output files** (same base name as the log, in `log/eat-diqkd-sep/ir_demo_keygen/`):

- `<log>_alice.npy` — Alice's own sifted key (effective positions; the AIR
  protocol never modifies her data).
- `<log>_bob_corrected.npy` — Bob's corrected key from the **last attempted
  stage** (effective, non-shortened positions only; the U-side estimate is
  recovered by re-encoding the decoded codeword), saved regardless of CRC
  success.

On a successful reconciliation the two files are bit-identical.

**Memory:** Bob builds **one** CA-SCL decoder per stage (~4.5 GB at $N=2^{20}$,
SCL$=8$) and frees it (`del` + `gc.collect()`) before the next stage, so peak
memory stays at ~2 decoders — do not pre-build all stage decoders.

**Verified** on `data/data4eat-sep/` (real BER$= 0.033728$): the demo succeeds
at stage 1 ($f = 1.05$) — assumed efficiency $1.050$, real efficiency $1.115$
(204,349 effective frozen bits + CRC-32 = 204,381 leaked bits; the demo's
efficiency counters also include the 1-bit success flag, which the ir-raw and
MC leakage counts do not); decode ≈92 s per stage at SCL$=8$ (committed demo log).

---

### 7. `qkd-air-demo-alice-chsh.ipynb` / `qkd-air-demo-bob-chsh.ipynb` — Interactive AIR Demo on Real CHSH Data

**Purpose:** Same interactive AIR demo as section 6, but for the **CHSH
Bell-test** codec ($N = 2^{15}$, SCL$= 32$, pure shortening). Alice encodes her
(AIR-flipped) CHSH bits and streams the CRC tag and the appended frozen bits
over HTTP; Bob applies the SNSPD + per-subgroup CHSH flips, feeds his BSC
LLRs directly to the CA-SCL decoder (which amplifies the known-zero padding
positions natively via `set_known_bits`).
The shortened codeword bits are zero by construction, so they are known to
both parties and never transmitted; the decoder hard-freezes those leaves.

**Defaults (same convention as `ir-raw-chsh.py`):**

| Parameter | Value | Note |
|---|---|---|
| Assumed QBER | $0.1825$ | $95\%$ Wald upper bound on the CHSH calibration BER (point $0.172920$, $\sigma = 0.004883$, upper $0.182492$), ceiled to $5 \times 10^{-4}$. Reliability file exists at exactly this BER (mu$=100$). |
| Stages | $f = 1.05, f = 1.10$ | configured efficiencies vs. assumed ideal |
| Final stage | union bound @ $99.999\%$ | minimal frozen bits for failure rate $\varepsilon = 10^{-5}$ |
| SCL / CRC | $32 / 32$ | |
| Trimming | $S = 12871, P = 0$ | pure shortening ($N = 2^{15} = 32768$) |

**Usage:**

1. Start **Bob's** notebook first (listens on port 8003, sends on 8012).
2. Run **Alice's** notebook (listens on port 8012, sends on 8003).
3. Check the logs in `log/eat-diqkd-sep/ir_demo_chsh/`:
   `qkd_air_demo_{alice,bob}_chsh_assumed_0.1825_*.txt`.

**Output files** (same base name as the log, in `log/eat-diqkd-sep/ir_demo_chsh/`):

- `<log>_alice.npy` — Alice's own sifted key (effective positions; the AIR
  protocol never modifies her data).
- `<log>_bob_corrected.npy` — Bob's corrected key from the **last attempted
  stage** (effective, non-shortened/non-punctured positions only; the U-side
  estimate is recovered by re-encoding the decoded codeword), saved
  regardless of CRC success.

On a successful reconciliation the two files are bit-identical.

**Memory:** ~0.5 GB per CA-SCL decoder at $N=2^{15}$, SCL$=32$. Bob builds one
decoder per stage and frees it before the next $\to$ peak ~1 GB. A stage
decode takes a few seconds.

**Verified** on `data/data4eat-sep/` (real BER$= 0.173795$): the demo
succeeds at stage 1 ($f = 1.05$) — assumed efficiency $1.050$, real
efficiency $1.080$ (14,289 effective frozen bits + CRC-32 = 14,321 leaked
bits; the demo's efficiency counters also include the 1-bit success flag,
which the ir-raw and MC leakage counts do not); decode ≈2.5 s per stage
at SCL$=32$ (committed demo log).

---

## Typical Workflow

### 1. Calibration arrives

```text
data/data4eat-sep/calibration-data-analysis-sep.py
  → S, QBER_E, Q, assumed-QBER upper bounds (ceil 5e-4)
  → Choose the assumed QBER of each codec; pass it to the IR / MC programs
```

### 2. MC validation (CHSH)

```text
python air-mc-chsh-calibration.py --num-samples 1000 --n-threads 8
```

### 3. MC validation (ZZ keygen)

```text
python air-mc-keygen-calibration.py --num-samples 10000 --n-threads 32
```

### 3b. Real-data demo (ZZ keygen, interactive)

```text
# Bob first (listens on 8003), then Alice — see section 6 above
jupyter notebook qkd-air-demo-bob-keygen.ipynb
jupyter notebook qkd-air-demo-alice-keygen.ipynb
```

### 3c. Real-data demo (CHSH, interactive)

```text
# Bob first (listens on 8003), then Alice — see section 7 above
jupyter notebook qkd-air-demo-bob-chsh.ipynb
jupyter notebook qkd-air-demo-alice-chsh.ipynb
```

---

## Directory Layout

```text
src/eat-sep/
├── README.md                           ← this file
├── ir-raw-chsh.py                      ← CHSH real-data AIR (py_aff3ct, N=2^15)
├── ir-raw-keygen.py                    ← ZZ keygen real-data AIR (py_aff3ct, N=2^20)
├── air-mc-chsh-calibration.py          ← CHSH MC simulation (py_aff3ct, N=2^15)
├── air-mc-keygen-calibration.py        ← ZZ keygen MC simulation (py_aff3ct, N=2^20)
├── qkd-air-demo-alice-keygen.ipynb     ← Alice-side AIR demo on real keygen data
├── qkd-air-demo-bob-keygen.ipynb       ← Bob-side AIR demo on real keygen data
├── qkd-air-demo-alice-chsh.ipynb       ← Alice-side AIR demo on real CHSH data
└── qkd-air-demo-bob-chsh.ipynb         ← Bob-side AIR demo on real CHSH data
```

Related directories:

```text
conf/polar/sp-bsc/         ← joint S+P polar code files (.pc + frozen_bits + positions)
data/data4eat-sep/         ← raw experimental data (.npy) + analysis programs
                             (diqkd-analysis-sep.py, calibration-data-analysis-sep.py)
log/eat-diqkd-sep/         ← runtime logs
src/utils.py               ← shared utilities (sp_positions, puncturer, etc.)
```

---

## Dependency Summary

| Program | py_aff3ct | data4eat-sep | conf/polar/sp-bsc | RAM (peak) |
|---|---|---|---|---|
| `data/data4eat-sep/calibration-data-analysis-sep.py` | no | no | no | < 1 GB |
| `ir-raw-chsh.py` | yes | yes | yes | ~1 GB (SCL=32) |
| `ir-raw-keygen.py` | yes | yes | yes | ~8 GB (SCL=8) |
| `air-mc-chsh-calibration.py` | yes | no² | yes | ~1.6 GB × `n_threads` (SCL=32) |
| `air-mc-keygen-calibration.py` | yes | no² | yes | ~16 GB × `n_threads` (SCL=8) |
| `qkd-air-demo-*-keygen.ipynb` | yes | yes | yes | ~9 GB (SCL=8, 2 decoders) |
| `qkd-air-demo-*-chsh.ipynb` | yes | yes | yes | ~1 GB (SCL=32, 1 decoder) |

² Uses synthetic BSC data generated by `Source_random_fast` + `BSC_digital_p`.

---

## Building py_aff3ct (the pre-built AFF3CT static library)

`submodules/py_aff3ct` links an **imported, pre-built** AFF3CT static library
whose file name embeds AFF3CT's `git describe`, e.g.
`lib/aff3ct/build/lib/libaff3ct-3.0.2-7-g3df3151.a`. Two consequences:

* `cmake --build submodules/py_aff3ct/build` does **not** rebuild AFF3CT; it
  recompiles only the py_aff3ct sources and re-links against the existing
  `.a`. Changes under `lib/aff3ct/` (for example the per-clone PRNG seed fix,
  aff3ct `90acd1198`) therefore never reach the Python bindings unless AFF3CT
  is rebuilt first.
* `submodules/py_aff3ct/cmake/Modules/aff3ct-config.cmake` records that file
  name. After a rebuild the name changes together with the version string, so
  the config file must be re-pointed; otherwise CMake keeps linking the stale
  library and the bindings silently keep the old behaviour.

Correct sequence after checking out the pinned submodules:

```bash
# 1. build AFF3CT itself
cmake --build submodules/py_aff3ct/lib/aff3ct/build -j 8

# 2. re-point the imported target at the freshly built library
ls submodules/py_aff3ct/lib/aff3ct/build/lib/        # note the new file name
sed -i 's|<old library name>|<new library name>|g' \
    submodules/py_aff3ct/cmake/Modules/aff3ct-config.cmake

# 3. reconfigure and re-link the bindings
cmake -S submodules/py_aff3ct -B submodules/py_aff3ct/build
cmake --build submodules/py_aff3ct/build -j 8
```

Sanity check: the freshly linked `.so` must have a different md5 from the
previous one, and `build/CMakeFiles/py_aff3ct-module-lib.dir/link.txt` must
reference the new `.a`.

---

## Known-Zero Padding Amplification (the `set_known_bits` decoder task)

Both codecs pad their sifted keys with the public constant 0 at ALL shortened
positions. The CA-SCL decoder amplifies the channel LLRs at these positions
internally (its `set_known_bits` task, a finite ×17 stand-in for the +∞
"known bit 0" LLR, computed with the exact float arithmetic of the former
Python-side shortener) and hard-constrains the corresponding codeword leaves
to 0. This is the shortening branch of BGL: the shortened codeword bits are
zero by construction, so they are known to both parties and never
transmitted. The ×17 factor is saturated (×100/×1000 give bit-identical
decodes) yet finite, avoiding float overflow in the min–sum propagation.

---

## Channel Reliability Files (`conf/polar/sp-bsc/`)

The repository ships the reliability files of the two codec configurations
used by all program defaults and by the committed logs:

| File | Used by |
|---|---|
| `n15_sp-bsc_ber0.1825_mu100_shortened12871_punctured0.pc` | all CHSH programs (default config) + committed CHSH demo logs |
| `frozen_bits_mu=100_N=15_p=0.1825_shortened=12871_punctured=0.txt` | union-bound final stage of all CHSH programs (default schedule, MC, demos) |
| `shortened_positions_N=15_S=12871.txt` | CHSH config |
| `n20_sp-bsc_ber0.0365_mu40_shortened187329_punctured0.pc` | all keygen programs (default config) + committed keygen demo/MC logs |
| `shortened_positions_N=20_S=187329.txt`, `punctured_positions_N=20_P=0.txt` | keygen config |

**Gzipped PE table.** The $N=2^{20}$ frozen-bit probability table
(`frozen_bits_mu=40_N=20_p=0.0365_*.txt`, ~100 MB uncompressed) exceeds
sensible repository sizes and is stored gzip'd. Before running
`ir-raw-keygen.py` with the default stage schedule, the keygen MC simulation,
or the keygen demo notebooks, unpack it once:

```bash
gunzip conf/polar/sp-bsc/frozen_bits_mu=40_N=20_p=0.0365_shortened=187329_punctured=0.txt.gz
```

**Other configurations.** Reliability files for any other ($N$, $S$, $P$, BER)
combination can be regenerated with
[polar-fbgen-rust](https://github.com/Shen-Cao/polar-fbgen-rust) (batch
method, `--num-shortened S --num-punctured P -p BER -N log2(N)`), then
converted to the `.pc` ordering file with
`python data/pe2idx.py -i <frozen_bits_...txt> -o <name.pc> -t sp-bsc -p <BER>`
(see `python data/pe2idx.py --help`).
