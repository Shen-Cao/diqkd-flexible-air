# diqkd-flexible-air

Information reconciliation (IR) for device-independent quantum key distribution (DIQKD) based on shortened polar codes

## Environment Configuration

One can use `pip install -r requirements_pip.txt` to install packages with pip or use `conda install --yes --file requirements_conda.txt` to install packages in a conda virtual environment. The latter is recommended because it contains the version of python and C++ compliers (gcc and gxx 15.1.0) that are verified to successfully compile the py_aff3ct library.

> It is possible that conda cannot install the required packages from the file even in a completely new environment.
> In this case, you can try installing GCC and G++ first `conda install -c conda-forge gcc=15.1.0 gxx=15.1.0`.
Then install python `conda install python=3.11.13` and lastly `pip install -r requirements_pip.txt`

## Library py_aff3ct

This project uses the py_aff3ct library to implement polar codec. You may use `git submodule update --init --recursive` to initialize and update all submodules. To compile the library, follow the instructions in [py_aff3ct Readme file](https://github.com/Shen-Cao/py_aff3ct/blob/master/README.md).

> Remark: One may meet error of `uint_8 does not name a type` during compilation (e.g. WSL2-Ubuntu-22.04, gcc 15.1.0, gxx 15.1.0). A simple solution is to add `#include <stdint.h>` to the begining of reported file(s).

## Information Reconciliation

`docs` contains jupyter notebooks of certain IR realization.

`src` contains python source codes.

`conf` stores configuration files such as polar code frozen vectors and LDPC code matrix.

`data` stores sifted key data collected from experiments, and a python program to generate channel reliability index.
