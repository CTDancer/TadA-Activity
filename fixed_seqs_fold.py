# Imports
import os
import time
import hydra
import py3Dmol
from lm_design_fold import Designer


def count_diff(a, b):
    assert len(a) == len(b)
    cnt = sum([1 if a[i] != b[i] else 0 for i in range(len(a))])
    return cnt


def sample(seed, path):
    seq = "SRLEEELRRRLTESGSETPGTSESATPESSGEVQLQESGGGLVQPGGSLRLSCTASGVTISALNAMAMGWYRQAPGERRVMVAAVSERGNAMYRESVQGRFTVTRDFTNKMVSLQMDNLKPEDTAVYYCHVLEDRVDSFHDYWGQGTQVTVSS"
    TASK = "fixedseqs_fold"
    iteration = 10000
    save_interval = 1

    # Load hydra config from config.yaml
    with hydra.initialize_config_module(config_module="conf"):
        cfg = hydra.compose(
            config_name="config_fold_3fingers", 
            overrides=[
                f"task={TASK}",
                f"seed={seed}",
                "seq_init_random=False",
                f'tasks.fixedseqs_fold.path={path}',
                f'tasks.fixedseqs_fold.save_interval={save_interval}',
                f'tasks.fixedseqs_fold.num_iter={iteration}'  # DEBUG - use a smaller number of iterations
            ])

    # Create a designer from configuration
    des = Designer(cfg, seq)

    # Run the designer
    start_time = time.time()
    des.run_from_cfg()
    print("finished after %s hours", (time.time() - start_time) / 3600)
    
    diff = count_diff(des.output_seq, des.init_seqs)
    print(f"Final seq has changed {diff} amino acids. \n {des.output_seq}")

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'a') as f:
        f.write(f'>sample_iter{iteration}\n')
        f.write(f'{des.output_seq}\n')

    for i in range(5):
        best_seq = des.best_seq[i]
        seq = best_seq[2]
        diff = count_diff(seq, des.init_seqs)
        print(f"Best seq (loss={best_seq[1]}) has changed {diff} amino acids, in step {best_seq[0]}. \n {seq}")
        with open(path, 'a') as f:
            f.write(f'>best_iter{best_seq[0]}_loss{round(best_seq[1], 2)}\n')
            f.write(f'{seq}\n')

if __name__ == '__main__':
    seed = 2
    path = f'output/3fingers_ESM5_fold_expclamp_seed{seed}.fasta'
    print(path)
    sample(seed, path)