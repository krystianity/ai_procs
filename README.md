# ai_procs

Automated I/O vector preparation for neuronal networks across multiple platforms.

## Install

`pip install https://github.com/krystianity/ai_procs/archive/master.zip`

## Use

```python
from ai_procs import aip

options = aip.create_options(
    ["FillMissing", "Normalize", "Categorify"],
    ["age", "thal"],
    "target",
    aip.MODEL_TYPES.PYTORCH.value,
)

# load dataset into df
metadata = aip.analyse_df(df, options)
df = aip.prepare_df(df, metadata)
# train model on df
aip.store_metadata(metadata, "./metadata.json")
# export model
```

## Using on other platforms

Importing models and metadata for other platforms.

* with C++ ` https://github.com/krystianity/ai_procs_native`

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