# diqkd-flexible-air

[简体中文](README_zh-CN.md) | [English](README.md)

> **请注意：** 本文件通过GLM-5驱动的[opencode](https://github.com/anomalyco/opencode)直接由[README.md](README.md)翻译而来。

基于截短极化码的设备无关量子密钥分发（DIQKD）信息协调（IR）

## 项目结构

```text
diqkd-flexible-air/
├── src/                          # Python 源代码
│   ├── qkd-air-demo-alice.py     # Alice 端演示程序
│   ├── qkd-air-demo-bob.py       # Bob 端演示程序
│   └── utils.py                  # 工具模块（编码器、解码器、HTTP 通信）
├── data/                         # 实验筛选密钥数据
│   ├── Alice_key.bin             # Alice 的筛选密钥（来自实验）
│   ├── Bob_key.bin               # Bob 的筛选密钥（来自实验）
│   └── pe2idx.py                 # 生成信道可靠性索引的脚本
├── conf/                         # 配置文件
│   └── polar/bsc/                # BSC 信道的极化码冻结比特向量
├── docs/                         # 文档和笔记本
│   └── qkd-air.ipynb             # Jupyter notebook 演示
├── log/                          # 运行日志
├── submodules/                   # Git 子模块
│   └── py_aff3ct/                # AFF3CT 库（Python 绑定）
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

本项目使用 py_aff3ct 库实现极化码编解码。可以使用 `git submodule update --init --recursive` 初始化并更新所有子模块。编译该库请遵循 [py_aff3ct README 文件](https://github.com/Shen-Cao/py_aff3ct/blob/master/README.md) 中的说明。

> 编译故障排除：如果在编译过程中遇到 `uint_8 does not name a type` 错误（例如 WSL2-Ubuntu-22.04，gcc 15.1.0，gxx 15.1.0），一个简单的解决方案是在报告的文件开头添加 `#include <stdint.h>`。

## 配置与使用

如需交互式演示，请参见 `docs/qkd-air.ipynb`。

`conf/` 存储编解码器的配置文件。

`data/` 存储从实验中收集的筛选密钥数据，以及一个用于生成信道可靠性索引的 Python 程序。

`src/` 文件夹中的 `.py` 文件模拟 Alice 和 Bob 在两台独立计算机上的协调过程。

**运行演示：**

**步骤 1**：在演示文件中配置网络地址：

- 编辑 `src/qkd-air-demo-alice.py` 和 `src/qkd-air-demo-bob.py`
- 适当设置 `sender_host`、`sender_port`、`receiver_host`、`receiver_port`
- 本地测试时，使用 `'localhost'`

**步骤 2**：首先运行 Bob 端（启动 HTTP 服务器）：

```bash
cd src
python qkd-air-demo-bob.py
```

**步骤 3**：在 Bob 的服务器启动后，运行 Alice 端：

```bash
cd src
python qkd-air-demo-alice.py
```

**步骤 4**：在 `log/demo/` 中查看完整的协调记录日志。

### 极化码配置

- 使用截短极化码处理任意长度的密钥
- 冻结比特位置由信道可靠性（BSC 信道，设计 BER）确定
- `conf/polar/bsc/` 中的配置文件遵循 `.pc` 格式

> 信道可靠性数据文件由论文 _How to Construct Polar Codes_ ([Arxiv 链接](https://arxiv.org/abs/1105.6164) 或 [IEEE TIT](https://ieeexplore.ieee.org/document/6557004)) 中的算法生成，该算法在 [PolarCodes-Encoding-Decoding-Construction](https://github.com/YuYongRun/PolarCodes-Encoding-Decoding-Construction) 中用 MATLAB 实现，在 [PolarCodeForQKD](https://github.com/cfxtby/PolarCodeForQKD) 中用 C++ 实现，并在 [polar-fbgen-rust](https://github.com/Shen-Cao/polar-fbgen-rust) 中用 Rust 重写。

## 数据

`data/` 中包含了来自 DIQKD 实验的筛选密钥数据：

- `Alice_key.bin` / `Bob_key.bin`：原始二进制筛选密钥
- 数据中的实际 QBER：约 3.6%

## 引用

如果您发现此仓库对您的工作有用，请引用我们的论文：

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

## 许可证

本项目采用 MIT 许可证 - 详情请参见 [LICENSE](LICENSE) 文件。

## 致谢

- [AFF3CT](https://aff3ct.github.io/) - 底层 FEC 库
- [py_aff3ct](https://github.com/Shen-Cao/py_aff3ct) - AFF3CT 的 Python 绑定
- [PolarCodeForQKD](https://github.com/cfxtby/PolarCodeForQKD)
- [PolarCodes-Encoding-Decoding-Construction](https://github.com/YuYongRun/PolarCodes-Encoding-Decoding-Construction)
