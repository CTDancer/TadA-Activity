import os
import hydra
from lm_design_fold import Designer
import pandas as pd
from tqdm import tqdm

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


def calc_loss_tsv(table_path, start, end):
    TASK = "fixedseqs_fold"
    num_recycles = 4

    # Load hydra config from config.yaml
    with hydra.initialize_config_module(config_module="conf"):
        cfg = hydra.compose(
            # config_name="config_fold_3fingers",
            config_name="config_covid19_3fingers",
            overrides=[
                f"task={TASK}",
                "debug=False",
                f'tasks.fixedseqs_fold.accept_reject.energy_cfg.num_recycles={num_recycles}',
                f'+tasks.fixedseqs_fold.focus_on_antigen={True}',
            ])

    seqs = load_table(table_path)
    end = min(len(seqs), end)
    print(f'[LOG] Aiming at ({start}-{end})/{len(seqs)}.')
    # Create a designer from configuration
    des = Designer(cfg)
    data = {'id':[], 'sequence':[], 'distance':[], 'pLDDT':[], 'loss':[]}
    for s in tqdm(range(start, end)):
        antibody, limit_range = seqs[s]
        des.cfg.tasks[des.cfg.task].limit_range = limit_range

        loss, log = des.calc_total_loss(antibody, cfg.tasks[cfg.task])
        data['id'].append(s)
        data['sequence'].append(antibody[16:])
        data['distance'].append(log['fold_loss'])
        data['pLDDT'].append(log['fold_conf'])
        data['loss'].append(log['total_loss'])

    # path_tsv = f'output_17k/Loss_{start}_{end}_fold-i4.tsv'
    path_tsv = f'output/covid/Loss_{start}_{end}_fold-i4.tsv'
    print('[LOG] Saving', path_tsv)

    df = pd.DataFrame(data)
    df.to_csv(path_tsv, sep="\t", index=False)


def calc_loss(seq):
    TASK = "fixedseqs_fold"
    conf_w = [-10]
    num_recycles = 4

    # Load hydra config from config.yaml
    with hydra.initialize_config_module(config_module="conf"):
        cfg = hydra.compose(
            config_name="config_fold_3fingers",
            overrides=[
                f"task={TASK}",
                "debug=False",
                f'tasks.fixedseqs_fold.accept_reject.energy_cfg.num_recycles={num_recycles}',
            ])

    # Create a designer from configuration
    des = Designer(cfg)
    loss, log = des.calc_total_loss(seq, cfg.tasks[cfg.task])
    print(loss, log)


if __name__ == '__main__':
    # seq = 'SGSETPGTSESATPESSGEVQLQESGGGLVQPGGSLRLSCTASGVTISALNAMAMGWYRQAPGERRVMVAAVSERGNAMYRESVQGRFTVTRDFTNKMVSLQMDNLKPEDTAVYYCHVLEDRVDSFHDYWGQGTQVTVSS'
    # calc_loss(seq)
    start = 8600
    end = 17200
    path = 'data/patent_sequence.tsv'
    calc_loss_tsv(path, start, end)