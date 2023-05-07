# Imports
import os
import time
import hydra
import py3Dmol
from lm_design import Designer


def count_diff(a, b):
    assert len(a) == len(b)
    cnt = sum([1 if a[i] != b[i] else 0 for i in range(len(a))])
    return cnt


def sample(seed, path):
    pdb_fn = os.getcwd() + '/data/ESM3.pdb'
    TASK = "fixedseqs"
    iteration = 100000
    save_interval = 100

    # Load hydra config from config.yaml
    with hydra.initialize_config_module(config_module="conf"):
        cfg = hydra.compose(
            config_name="config_seqs_3fingers", 
            overrides=[
                f"task={TASK}", 
                f"seed={seed}", 
                f"pdb_fn={pdb_fn}",
                "seq_init_random=False",
                f'tasks.fixedseqs.path={path}',
                f'tasks.fixedseqs.save_interval={save_interval}',
                f'tasks.fixedseqs.num_iter={iteration}'  # DEBUG - use a smaller number of iterations
            ])

    # Create a designer from configuration
    des = Designer(cfg, pdb_fn)

    # Run the designer
    start_time = time.time()
    des.run_from_cfg()
    print("finished after %s hours", (time.time() - start_time) / 3600)
    
    diff = count_diff(des.output_seq, des.wt_seq_raw)
    print(f"Output seq has changed {diff} amino acids. \n {des.output_seq}")
    with open(path, 'a') as f:
        f.write(f'>sample_iter{iteration}\n')
        f.write(f'{des.output_seq}\n')

if __name__ == '__main__':
    seed = 0
    path = f'output/mute_3fingers_ESM3_seed{seed}.fasta'
    print(path)
    sample(seed, path)