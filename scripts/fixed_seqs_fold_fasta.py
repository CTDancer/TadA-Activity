import os
import hydra
from lm_design_fold import Designer
import pandas as pd


def count_diff(a, b):
    assert len(a) == len(b)
    cnt = sum([1 if a[i] != b[i] else 0 for i in range(len(a))])
    return cnt


def load_table(path):
    linker = 'SGSETPGTSESATPES'

    antibody = []
    tb = pd.read_csv(path, sep='\t')
    for i in range(len(tb)):
        ab = linker + tb.iloc[i, 0]
        hand = [[s := ab.index(tb.iloc[i, j]), s + len(tb.iloc[i, j])] for j in range(1, 4)]
        assert hand[2][0] >= hand[1][1] and hand[1][0] >= hand[0][1] and hand[0][0]>=16
        antibody.append([ab, hand])
    return antibody


def sample(start, end, seed):
    TASK = "fixedseqs_fold"
    iteration = 10000
    save_interval = 1
    conf_nonlinear = ['relu', 'leakyrelu', '2', '3', None][4]
    conf_w = [-100]
    keep_best = 10
    num_recycles = 4
    temperature = 0.01
    conf_w_str = 'conf' if not conf_nonlinear else conf_nonlinear
    path = f'output/17k/fold-i{num_recycles}_I-{iteration}_{conf_w_str}{conf_w[0]}_T{temperature}_seed{seed}.fasta'
    print(path)
    pdb_dir = ''
    print(pdb_dir)
    table_path = 'data/patent_sequence.tsv'

    # Load hydra config from config.yaml
    with hydra.initialize_config_module(config_module="conf"):
        cfg = hydra.compose(
            # config_name="config_fold_3fingers",
            config_name="config_covid19_3fingers",
            overrides=[
                f"task={TASK}",
                f"seed={seed}",
                "debug=False",
                "seq_init_random=False",
                f"tasks.fixedseqs_fold.keep_best={keep_best}",
                f'tasks.fixedseqs_fold.path={path}',
                f'tasks.fixedseqs_fold.pdb_dir={pdb_dir}',
                f'tasks.fixedseqs_fold.save_interval={save_interval}',
                f'tasks.fixedseqs_fold.accept_reject.energy_cfg.conf_w={conf_w}',
                f'tasks.fixedseqs_fold.accept_reject.energy_cfg.conf_nonlinear={conf_nonlinear}',
                f'tasks.fixedseqs_fold.accept_reject.energy_cfg.num_recycles={num_recycles}',
                f'tasks.fixedseqs_fold.accept_reject.temperature.initial={temperature}',
                f'tasks.fixedseqs_fold.num_iter={iteration}'  # DEBUG - use a smaller number of iterations
            ])

    seqs = load_table(table_path)
    end = min(end, len(seqs))
    print(f'[LOG] Aiming at ({start}-{end})/{len(seqs)}.')
    # Create a designer from configuration
    des = Designer(cfg)
    for s in range(start, end):
        antibody, limit_range = seqs[s]
        des.x_seqs = antibody
        des.init_seqs = des.x_seqs
        des.cfg.tasks[des.cfg.task].limit_range = limit_range
        path = f'output_17k/best_loss_init_covid/{s}_fold-i{num_recycles}_I-{iteration}_{conf_w_str}{conf_w[0]}_T{temperature}_seed{seed}.fasta'
        des.cfg.tasks[des.cfg.task].path = path
        print('[LOG]', path)

        # Run the designer
        des.run_from_cfg()

        diff = count_diff(des.output_seq, des.init_seqs)
        # print(f"Final seq has changed {diff} amino acids. \n {des.output_seq}")

        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        with open(path, 'a') as f:
            f.write(f'>sample_iter{iteration}\n')
            f.write(f'{des.output_seq}\n')

        des.best_seq.append(des.origin_seq)
        for i in range(keep_best):
            best_seq = des.best_seq[i]
            seq = best_seq[2]
            diff = count_diff(seq, des.init_seqs)
            # print(f"Best seq (loss={best_seq[1]}) has changed {diff} amino acids, in step {best_seq[0]}. \n {seq}")
            with open(path, 'a') as f:
                f.write(f'>best_iter{best_seq[0]}_loss{round(best_seq[1], 2)}_{best_seq[3]}\n')
                f.write(f'{seq}\n')

if __name__ == '__main__':
    seed = 1
    # lists = [15542, 10608, 11667, 6415, 17039, 11487, 13097, 8343, 66, 6389]
    lists = [9671, 8805]
    start = lists[1]
    end = start + 1
    # start = 0
    # end = 17200
    sample(start, end, seed)