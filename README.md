# diqkd-flexible-air

[简体中文](README_zh-CN.md) | [English](README.md)

Separate information reconciliation (IR) for device-independent quantum key distribution (DIQKD) with shortened polar codes

## Overview

This repository contains the **separate-IR** information reconciliation
pipeline for DIQKD. In the separate-IR scheme, CHSH Bell-test rounds
(~19.9k pairs, BER ≈ 17%) and ZZ key-generation rounds (~861k pairs,
BER ≈ 3.4%) are reconciled by **two independent polar codes**, each matched to
its own BER regime:

| Codec | Data | N | L | Assumed BER | Trimming |
|---|---|---|---|---|---|
| CHSH | ~19.9k Bell-test rounds | 2¹⁵ = 32,768 | 19,897 | 0.1825 | S=12871, P=0 (pure shortening) |
| ZZ keygen | ~861k Z⊗Z key-gen rounds | 2²⁰ = 1,048,576 | 861,247 | 0.0365 | S=187329, P=0 (pure shortening) |

Both codecs use the Bioglio–Gabry–Land (BGL) construction for rate
compatibility, decoded with CRC-aided successive cancellation list (CA-SCL)
decoding under the appending information reconciliation (AIR) protocol. In
the released configuration both are purely shortened (P = 0); the repository
also documents why the punctured variant is incompatible with this decoder;
see [src/eat-sep/study-puncturing/README.md](src/eat-sep/study-puncturing/README.md).

The full program documentation and the typical workflow live in
[`src/eat-sep/README.md`](src/eat-sep/README.md).

Besides the separate-IR pipeline, the repository also retains the original
**ion–ion AIR demo** programs corresponding to the Nature 2026 experiment
(`src/ion2ion/`) — see [The Ion–ion Demo](#the-ion-ion-demo-nature-2026-experiment).

## Project Structure

```text
diqkd-flexible-air/
├── src/
│   ├── eat-sep/                  # Separate-IR programs:
│   │   ├── ir-raw-chsh.py        # CHSH real-data AIR pipeline (N=2^15)
│   │   ├── ir-raw-keygen.py      # ZZ keygen real-data AIR pipeline (N=2^20)
│   │   ├── air-mc-chsh-calibration.py         # CHSH Monte-Carlo validation
│   │   ├── air-mc-keygen-calibration.py       # ZZ keygen Monte-Carlo validation
│   │   └── qkd-air-demo-*.ipynb  # interactive Alice/Bob AIR demos (CHSH & keygen)
│   ├── ion2ion/                  # Original ion–ion AIR demo (Nature 2026 experiment):
│   │   ├── qkd-air-demo-alice.py # Alice-side demo program
│   │   ├── qkd-air-demo-bob.py   # Bob-side demo program
│   │   ├── qkd-air.ipynb         # Jupyter notebook demonstration
│   │   └── images/               # Figure referenced by the notebook
│   └── utils.py                  # Shared py_aff3ct building blocks (modules, polar helpers, union bound)
├── data/
│   ├── data4eat-sep/             # Experimental dataset (.npy) + analysis programs
│   │                             # (diqkd-analysis-sep.py, calibration-data-analysis-sep.py)
│   ├── ion2ion/                  # Sifted keys from the Nature 2026 experiment
│   └── pe2idx.py                 # PE table → channel reliability index conversion
├── conf/
│   └── polar/
│       ├── sp-bsc/               # Joint S+P reliability files (separate-IR codecs)
│       └── bsc/                  # Reliability files for the ion–ion demo (N=2^17)
├── log/
│   ├── eat-diqkd-sep/            # Logs of the separate-IR demo and Monte-Carlo results
│   └── demo/                     # Logs of the ion–ion demo
├── submodules/
│   └── py_aff3ct/                # AFF3CT library (Python bindings, custom polar modules)
├── requirements_pip.txt          # pip dependencies
└── requirements_conda.txt        # conda dependencies
```

## Environment Setup

### Option 1: Conda

The conda environment includes verified Python and C++ compiler versions:

```bash
# Create and activate conda environment
conda create -n diqkd
conda activate diqkd

conda install --yes --file requirements_conda.txt
```

### Option 2: Conda and pip

If conda cannot install the required packages from the file even in a completely new environment, then you can try to install GCC/G++ compilers first, then use `pip`:

```bash
# Create and activate conda environment
conda create -n diqkd
conda activate diqkd
# Install GCC/G++ compilers (required for py_aff3ct)
conda install -c conda-forge gcc=15.1.0 gxx=15.1.0

# Install Python
conda install python=3.11.13

# Install Python dependencies
pip install -r requirements_pip.txt
```

> **Note**: If conda fails to install packages, try installing GCC/G++ first, then Python, and finally the pip requirements.

## Building py_aff3ct

This project uses the py_aff3ct library to implement the polar codecs. Use
`git submodule update --init --recursive` to initialize and update all
submodules. To compile the library, follow the instructions in the
[py_aff3ct Readme file](https://github.com/Shen-Cao/py_aff3ct/blob/master/README.md).
The programs expect the built library under `submodules/py_aff3ct/build/lib`.

> Compilation Troubleshooting: If you encounter `uint_8 does not name a type` error during compilation (e.g. WSL2-Ubuntu-22.04, gcc 15.1.0, gxx 15.1.0). A simple solution is to add `#include <stdint.h>` to the beginning of the reported file(s).

## Quick Start

**Step 1**: Unpack the gzipped channel reliability table (needed by the
keygen programs that use the union bound — `ir-raw-keygen.py` with the
default stage schedule, the MC simulation, and the demo notebooks):

```bash
gunzip conf/polar/sp-bsc/frozen_bits_mu=40_N=20_p=0.0365_shortened=187329_punctured=0.txt.gz
```

**Step 2**: Smoke-test the calibration analysis (numpy only, no py_aff3ct):

```bash
python data/data4eat-sep/calibration-data-analysis-sep.py
python data/data4eat-sep/diqkd-analysis-sep.py
```

**Step 3**: Run a real-data reconciliation or a Monte-Carlo validation —
see [`src/eat-sep/README.md`](src/eat-sep/README.md) for the complete
workflow, CLI reference, memory requirements, and verified results.

## The Ion–ion Demo (Nature 2026 Experiment)

`src/ion2ion/` contains the original information reconciliation demo of the
ion–ion entanglement experiment reported in the Nature 2026 paper (see
[Citation](#citation)):

- `qkd-air-demo-alice.py` / `qkd-air-demo-bob.py` — the two-party AIR demo
  over HTTP on the experimental sifted keys `data/ion2ion/{Alice,Bob}_key.bin`
  (shortened polar code at N = 2¹⁷, assumed QBER 0.0375, SCL = 16)
- `qkd-air.ipynb` — a notebook walking through the whole scheme on the same
  data (assumed QBER 0.0345)

**To run the two-party demo:**

1. Edit `sender_host` / `sender_port` / `receiver_host` / `receiver_port` in
   both files (use `'localhost'` for local testing; the committed files still
   carry the IP addresses of the original run).
2. Run Bob's side first (it starts the HTTP server), then Alice's side:

```bash
cd src/ion2ion
python qkd-air-demo-bob.py     # terminal 1
python qkd-air-demo-alice.py   # terminal 2
```

3. Check the reconciliation records in `log/demo/` (the committed logs there
   were produced the same way).

### Polar Code Configuration

- The BGL construction (shortening and puncturing) handles arbitrary block
  lengths between powers of two; both released codecs use pure shortening
  ($S = N - L$, $P = 0$)
- Frozen bit positions are determined by channel reliability (BSC with the
  assumed design BER), excluding the shortened (and, when present, punctured)
  positions
- Configuration files in `conf/polar/sp-bsc/` follow the `.pc` format

> The channel reliability data files were generated by algorithm from the paper _How to Construct Polar Codes_ ([Arxiv link](https://arxiv.org/abs/1105.6164) or [IEEE TIT](https://ieeexplore.ieee.org/document/6557004)), which is implemented with MATLAB in [PolarCodes-Encoding-Decoding-Construction](https://github.com/YuYongRun/PolarCodes-Encoding-Decoding-Construction), with C++ in [PolarCodeForQKD](https://github.com/cfxtby/PolarCodeForQKD) and rewritten by Rust in [polar-fbgen-rust](https://github.com/Shen-Cao/polar-fbgen-rust).

## Data

`data/data4eat-sep/` contains the DIQKD experimental dataset used by the
real-data programs (881,144 rounds in total):

- 19,897 CHSH Bell-test rounds (BER after 3-step flip ≈ 17.38%)
- 861,247 Z⊗Z key-generation rounds (BER after flip ≈ 3.37%)
- `diqkd-analysis-sep.py` reproduces these statistics from the raw `.npy`
  files

`data/ion2ion/` contains the sifted keys of the Nature 2026 experiment
used by the ion–ion demo (real QBER ≈ 3.6%).

`log/eat-diqkd-sep/` contains the committed logs of the separate-IR demos
(`ir_demo_chsh/`, `ir_demo_keygen/`) and Monte-Carlo validations
(`mc_chsh_n10000/`, `mc_keygen_n10000/`); `log/demo/` contains the logs of
the ion–ion demo.

The committed Monte-Carlo run configurations (including the RNG seed) are
documented in `src/eat-sep/README.md`; the CHSH 10,000-frame run was produced
with `--seed 39`.

## Citation

If you find this repo useful for your work, please cite our paper:

```bibtex
@article{
    nature_2026,
    title={Long-lived remote ion–ion entanglement for scalable quantum repeaters},
    rights={2026 The Author(s), under exclusive licence to Springer Nature Limited},
    ISSN={1476-4687},
    DOI={10.1038/s41586-026-10177-4},
    journal={Nature},
    publisher={Nature Publishing Group},
    author={Liu, Wen-Zhao and Zhou, Ya-Bin and Chen, Jiu-Peng and Wang, Bin and Teng, Ao and Han, Xiao-Wen and Liu, Guang-Cheng and Zhang, Zhi-Jiong and Yang, Yi and Liu, Feng-Guang and Xue, Chao-Hui and Yang, Bo-Wen and Yang, Jin and Zeng, Chao and Pan, Du-Ruo and Zheng, Ming-Yang and Zhang, Xingjian and Cao, Shen and Zhen, Yi-Zheng and Xiao, You and Li, Hao and You, Lixing and Ma, Xiongfeng and Zhao, Qi and Xu, Feihu and Wang, Ye and Wan, Yong and Zhang, Qiang and Pan, Jian-Wei},
    year={2026},
    month=feb,
    pages={1–7},
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [AFF3CT](https://aff3ct.github.io/) - The underlying FEC library
- [py_aff3ct](https://github.com/Shen-Cao/py_aff3ct) - Python bindings for AFF3CT
- [PolarCodeForQKD](https://github.com/cfxtby/PolarCodeForQKD)
- [PolarCodes-Encoding-Decoding-Construction](https://github.com/YuYongRun/PolarCodes-Encoding-Decoding-Construction)
