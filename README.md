# Introduction
This repository is based on [lm-design](https://github.com/facebookresearch/esm/tree/main/examples/lm-design).


# Installation

```bash
conda create -n AIGP_gj python=3.8 -y
conda activate AIGP_gj
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

# Usage

```bash
# DIY your file
python fixed_positions.py

# Use the shell args
python -m lm_design task=fixedseqs pdb_fn=data/ESM3.pdb --seed 0
```

Other, more advanced configurations can be observed at [config_seqs.yaml](conf/config_seqs.yaml)