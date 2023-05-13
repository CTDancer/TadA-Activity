import os
import hydra
from lm_design_fold import Designer


def fasta2seqs(path):
    pass


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
    seq = 'SGSETPGTSESATPESSGEVQLQESGGGLVQPGGSLRLSCTASGVTISALNAMAMGWYRQAPGERRVMVAAVSERGNAMYRESVQGRFTVTRDFTNKMVSLQMDNLKPEDTAVYYCHVLEDRVDSFHDYWGQGTQVTVSS'
    calc_loss(seq)