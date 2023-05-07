# Introduction
This repository is based on [lm-design](https://github.com/facebookresearch/esm/tree/main/examples/lm-design).


# Installation

```bash
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