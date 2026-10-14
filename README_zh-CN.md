# diqkd-flexible-air

[简体中文](README_zh-CN.md) | [English](README.md)

> **请注意：** 本文件通过 DeepSeek V4.1 Flash 驱动的 [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) 直接由 [README.md](README.md) 翻译而来。
>
> [![](https://img.shields.io/badge/powered_by-dsh-4D6BFE?style=flat-square&logo=deepseek&logoColor=white)](https://github.com/deepseek-ai/deepseek-harness)

基于截短极化码的设备无关量子密钥分发（DIQKD）独立信息协调（IR）

## 概述

本仓库包含 DIQKD 的**独立协调（separate-IR）**信息协调管线。在独立 IR
方案中，CHSH Bell 检验轮（约 1.99 万对，BER ≈ 17%）与 ZZ 密钥生成轮
（约 86.1 万对，BER ≈ 3.4%）由**两套独立的极化码**分别协调，每套码各自
匹配其 BER 区间：

| 编解码器 | 数据 | N | L | 假定 BER | 修剪方式 |
|---|---|---|---|---|---|
| CHSH | ~19.9k Bell 检验轮 | 2¹⁵ = 32,768 | 19,897 | 0.1825 | S=12871, P=0（纯截短） |
| ZZ keygen | ~861k Z⊗Z 密钥生成轮 | 2²⁰ = 1,048,576 | 861,247 | 0.0365 | S=187329, P=0（纯截短） |

两套码均采用 Bioglio–Gabry–Land（BGL）构造实现率兼容，在追加式信息协调
（AIR）协议下使用 CRC 辅助连续消去列表（CA-SCL）译码。本次发布的两套码
均为纯截短（P = 0）；仓库同时记录了打孔变体与本译码器不兼容的原因，见
[src/eat-sep/study-puncturing/README.md](src/eat-sep/study-puncturing/README.md)。

完整的程序文档与典型工作流见
[`src/eat-sep/README.md`](src/eat-sep/README.md)。

除独立 IR 管线外，本仓库还保留了与 Nature 2026 实验相对应的原始
**离子–离子 AIR 演示**程序（`src/ion2ion/`），见
[离子–离子演示](#离子离子演示nature-2026-实验)。

## 项目结构

```text
diqkd-flexible-air/
├── src/
│   ├── eat-sep/                  # 独立 IR 程序：
│   │   ├── ir-raw-chsh.py        # CHSH 实数据 AIR 管线（N=2^15）
│   │   ├── ir-raw-keygen.py      # ZZ 密钥生成实数据 AIR 管线（N=2^20）
│   │   ├── air-mc-chsh-calibration.py         # CHSH 蒙特卡洛验证
│   │   ├── air-mc-keygen-calibration.py       # ZZ 密钥生成蒙特卡洛验证
│   │   └── qkd-air-demo-*.ipynb  # Alice/Bob 交互式 AIR 演示（CHSH 与 keygen）
│   ├── ion2ion/                  # 原始离子–离子 AIR 演示（Nature 2026 实验）：
│   │   ├── qkd-air-demo-alice.py # Alice 端演示程序
│   │   ├── qkd-air-demo-bob.py   # Bob 端演示程序
│   │   ├── qkd-air.ipynb         # Jupyter notebook 演示
│   │   └── images/               # notebook 引用的插图
│   └── utils.py                  # 共享 py_aff3ct 构件（自定义模块、极化码辅助函数、union bound）
├── data/
│   ├── data4eat-sep/             # 实验数据集（.npy）+ 数据分析程序
│   │                             #（diqkd-analysis-sep.py、calibration-data-analysis-sep.py）
│   ├── ion2ion/                  # Nature 2026 实验的筛选密钥
│   └── pe2idx.py                 # PE 表 → 信道可靠性索引转换工具
├── conf/
│   └── polar/
│       ├── sp-bsc/               # 联合 S+P 可靠性文件（独立 IR 编解码器）
│       └── bsc/                  # 离子–离子演示的可靠性文件（N=2^17）
├── log/
│   ├── eat-diqkd-sep/            # 独立 IR 演示与蒙特卡洛结果日志
│   └── demo/                     # 离子–离子演示日志
├── submodules/
│   └── py_aff3ct/                # AFF3CT 库（Python 绑定，含自定义极化码模块）
├── requirements_pip.txt          # pip 依赖
└── requirements_conda.txt        # conda 依赖
```

## 环境配置

### 方式一：Conda

Conda 环境包含经过验证的 Python 和 C++ 编译器版本：

```bash
# 创建并激活 conda 环境
conda create -n diqkd
conda activate diqkd

conda install --yes --file requirements_conda.txt
```

### 方式二：Conda 和 pip

如果 conda 即使在全新的环境中也无法从文件安装所需的包，可以先尝试安装 GCC/G++ 编译器，然后使用 `pip`：

```bash
# 创建并激活 conda 环境
conda create -n diqkd
conda activate diqkd
# 安装 GCC/G++ 编译器（py_aff3ct 需要）
conda install -c conda-forge gcc=15.1.0 gxx=15.1.0

# 安装 Python
conda install python=3.11.13

# 安装 Python 依赖
pip install -r requirements_pip.txt
```

> **注意**：如果 conda 安装包失败，请尝试先安装 GCC/G++，然后安装 Python，最后安装 pip 依赖。

## 编译 py_aff3ct

本项目使用 py_aff3ct 库实现极化码编解码。可以使用 `git submodule update --init --recursive` 初始化并更新所有子模块。编译该库请遵循 [py_aff3ct README 文件](https://github.com/Shen-Cao/py_aff3ct/blob/master/README.md) 中的说明。各程序默认在 `submodules/py_aff3ct/build/lib` 下寻找编译产物。

> 编译故障排除：如果在编译过程中遇到 `uint_8 does not name a type` 错误（例如 WSL2-Ubuntu-22.04, gcc 15.1.0, gxx 15.1.0），一个简单的解决方案是在报告的文件开头添加 `#include <stdint.h>`。

## 快速开始

**步骤 1**：解压 gzip 压缩的信道可靠性表（使用 union bound 的 keygen 程序
需要——默认效率分级的 `ir-raw-keygen.py`、蒙特卡洛仿真与演示笔记本）：

```bash
gunzip conf/polar/sp-bsc/frozen_bits_mu=40_N=20_p=0.0365_shortened=187329_punctured=0.txt.gz
```

**步骤 2**：冒烟测试校准分析（仅依赖 numpy，无需 py_aff3ct）：

```bash
python data/data4eat-sep/calibration-data-analysis-sep.py
python data/data4eat-sep/diqkd-analysis-sep.py
```

**步骤 3**：运行实数据协调或蒙特卡洛验证——完整的工作流、CLI 参考、
内存需求与经验证结果见 [`src/eat-sep/README.md`](src/eat-sep/README.md)。

## 离子–离子演示（Nature 2026 实验）

`src/ion2ion/` 包含 Nature 2026 论文（见[引用](#引用)）所报道的离子–离子
纠缠实验的原始信息协调演示：

- `qkd-air-demo-alice.py` / `qkd-air-demo-bob.py` —— 基于实验筛选密钥
  `data/ion2ion/{Alice,Bob}_key.bin` 的双机 HTTP AIR 演示
  （截短极化码 N = 2¹⁷，假定 QBER 0.0375，SCL = 16）
- `qkd-air.ipynb` —— 在同一数据上完整走通整个方案的 notebook
  （假定 QBER 0.0345）

**运行双机演示：**

1. 在两个文件中编辑 `sender_host` / `sender_port` / `receiver_host` /
   `receiver_port`（本地测试用 `'localhost'`；已提交文件中保留的是原始
   运行时的 IP 地址）。
2. 先运行 Bob 端（启动 HTTP 服务器），再运行 Alice 端：

```bash
cd src/ion2ion
python qkd-air-demo-bob.py     # 终端 1
python qkd-air-demo-alice.py   # 终端 2
```

3. 在 `log/demo/` 查看完整的协调记录（其中已提交的日志即以此方式产生）。

### 极化码配置

- BGL 构造（截短与打孔）可处理 2 的幂之间的任意码长；本次发布的两套码均
  采用纯截短（$S = N - L$，$P = 0$）
- 冻结比特位置由信道可靠性（采用假定设计 BER 的 BSC 信道）确定，并排除
  截短（以及存在时的打孔）位置
- `conf/polar/sp-bsc/` 中的配置文件遵循 `.pc` 格式

> 信道可靠性数据文件由论文 _How to Construct Polar Codes_ ([Arxiv 链接](https://arxiv.org/abs/1105.6164) 或 [IEEE TIT](https://ieeexplore.ieee.org/document/6557004)) 中的算法生成，该算法在 [PolarCodes-Encoding-Decoding-Construction](https://github.com/YuYongRun/PolarCodes-Encoding-Decoding-Construction) 中用 MATLAB 实现，在 [PolarCodeForQKD](https://github.com/cfxtby/PolarCodeForQKD) 中用 C++ 实现，并在 [polar-fbgen-rust](https://github.com/Shen-Cao/polar-fbgen-rust) 中用 Rust 重写。

## 数据

`data/data4eat-sep/` 包含实数据程序所用的 DIQKD 实验数据集（共
881,144 轮）：

- 19,897 个 CHSH Bell 检验轮（三步翻转后 BER ≈ 17.38%）
- 861,247 个 Z⊗Z 密钥生成轮（翻转后 BER ≈ 3.37%）
- `diqkd-analysis-sep.py` 可从原始 `.npy` 文件复现上述统计量

`data/ion2ion/` 包含 Nature 2026 实验的筛选密钥（实际 QBER ≈ 3.6%），
供离子–离子演示使用。

`log/eat-diqkd-sep/` 包含已提交的独立 IR 演示日志（`ir_demo_chsh/`、
`ir_demo_keygen/`）与蒙特卡洛验证日志（`mc_chsh_n10000/`、
`mc_keygen_n10000/`）；`log/demo/` 为离子–离子演示日志。

已提交的蒙特卡洛运行配置（含 RNG 种子）记录在 `src/eat-sep/README.md`；CHSH 一万帧运行使用 `--seed 39`。

## 引用

如果您发现此仓库对您的工作有用，请引用我们的论文：

### 采用量子互补性（QC）安全性分析的 DIQKD 信息协调（AIR）

```bibtex
@article{
    nature_2026,
    title={Long-lived remote ion–ion entanglement for scalable quantum repeaters},
    rights={2026 The Author(s), under exclusive licence to Springer Nature Limited},
    ISSN={1476-4687},
    DOI={10.1038/s41586-026-10177-4},
    journal={Nature},
    volume={652},
    number={8108},
    publisher={Nature Publishing Group},
    author={Liu, Wen-Zhao and Zhou, Ya-Bin and Chen, Jiu-Peng and Wang, Bin and Teng, Ao and Han, Xiao-Wen and Liu, Guang-Cheng and Zhang, Zhi-Jiong and Yang, Yi and Liu, Feng-Guang and Xue, Chao-Hui and Yang, Bo-Wen and Yang, Jin and Zeng, Chao and Pan, Du-Ruo and Zheng, Ming-Yang and Zhang, Xingjian and Cao, Shen and Zhen, Yi-Zheng and Xiao, You and Li, Hao and You, Lixing and Ma, Xiongfeng and Zhao, Qi and Xu, Feihu and Wang, Ye and Wan, Yong and Zhang, Qiang and Pan, Jian-Wei},
    year={2026},
    pages={51–57},
}
```

<!-- ### 采用 EAT 安全性分析的 DIQKD 独立信息协调（separate IR） -->

<!-- 待发表 -->

## 许可证

本项目采用 MIT 许可证 - 详情请参见 [LICENSE](LICENSE) 文件。

## 致谢

- [AFF3CT](https://aff3ct.github.io/) - 底层 FEC 库
- [py_aff3ct](https://github.com/Shen-Cao/py_aff3ct) - AFF3CT 的 Python 绑定
- [PolarCodeForQKD](https://github.com/cfxtby/PolarCodeForQKD)
- [PolarCodes-Encoding-Decoding-Construction](https://github.com/YuYongRun/PolarCodes-Encoding-Decoding-Construction)
