import os
import time
import hydra
from lm_design_fold import Designer


def count_diff(a, b):
    assert len(a) == len(b)
    cnt = sum([1 if a[i] != b[i] else 0 for i in range(len(a))])
    return cnt


def sample(seed):
    TASK = "fixedseqs_fold"
    iteration = 1000
    save_interval = [1, 'best'][1]
    conf_nonlinear = ['relu', 'leakyrelu', '2', '3', None][4]
    conf_w = [-10]
    keep_best = 1
    num_recycles = 4
    temperature = 0.01
    focus_on_antigen = False
    conf_w_str = 'conf' if not conf_nonlinear else conf_nonlinear
    path = f'output/ESM3_fold-i{num_recycles}_I-{iteration}_{conf_w_str}{conf_w[0]}_T{temperature}_seed{seed}.fasta'
    print(path)
    pdb_dir = f'output/pdb_ESM3_fold-i{num_recycles}_I-{iteration}_conf-{conf_w_str}_T{temperature}_seed{seed}'
    # pdb_dir = ''
    print(pdb_dir)

    # Load hydra config from config.yaml
    with hydra.initialize_config_module(config_module="conf"):
        cfg = hydra.compose(
            config_name="config_fold_3fingers",
            overrides=[
                f"task={TASK}",
                f"seed={seed}",
                "debug=False",
                "seq_init_random=False",
                f"tasks.fixedseqs_fold.keep_best={keep_best}",
                f'tasks.fixedseqs_fold.path={path}',
                f'tasks.fixedseqs_fold.pdb_dir={pdb_dir}',
                f'tasks.fixedseqs_fold.save_interval={save_interval}',
                f'+tasks.fixedseqs_fold.focus_on_antigen={focus_on_antigen}',
                f'tasks.fixedseqs_fold.accept_reject.energy_cfg.conf_w={conf_w}',
                f'tasks.fixedseqs_fold.accept_reject.energy_cfg.conf_nonlinear={conf_nonlinear}',
                f'tasks.fixedseqs_fold.accept_reject.energy_cfg.num_recycles={num_recycles}',
                f'tasks.fixedseqs_fold.accept_reject.temperature.initial={temperature}',
                f'tasks.fixedseqs_fold.num_iter={iteration}'  # DEBUG - use a smaller number of iterations
            ])

    # Create a designer from configuration
    des = Designer(cfg)

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

    for i in range(keep_best):
        best_seq = des.best_seq[i]
        seq = best_seq[2]
        diff = count_diff(seq, des.init_seqs)
        print(f"Best seq (loss={best_seq[1]}) has changed {diff} amino acids, in step {best_seq[0]}. \n {seq}")
        with open(path, 'a') as f:
            f.write(f'>best_iter{best_seq[0]}_loss{round(best_seq[1], 2)}_{best_seq[3]}\n')
            f.write(f'{seq}\n')

if __name__ == '__main__':
    seed = 1
    sample(seed)