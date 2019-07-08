# ai_procs

## Install

`pip install git@github.com:krystianity/ai_procs.git`

## Use

TBD

## Develop locally

### Setup

```bash
brew install python3
open https://www.anaconda.com/distribution/#macos
# download the installer for python 3.7 and install it
nano ~/.zshrc
export PATH=/anaconda3/bin:$PATH
# restart shell
conda init zsh
# restart shell
```

### Creating the environment

```bash
cd pt-model-workflow
conda create -n aiprocs python=3.6 pip
conda activate aiprocs
conda install -c numpy pandas
# conda deactivate # in case of leaving
```