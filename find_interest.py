import os
import hydra
from lm_design_fold import Designer
import pandas as pd
from tqdm import tqdm


def find_interest(antibody, antigen, limit_range, len_insterest, mode='brute'):
    TASK = "fixedseqs_fold"
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
    # import pdb; pdb.set_trace()
    losses = des.find_interest(antigen, antibody, len_insterest, limit_range, 4, mode)
    import pdb; pdb.set_trace()
    print(losses)

if __name__ == '__main__':
    antigen = 'WRQTWSGPGPDRKAAVSHWQQVQRDMFTLEDTLLGYLADDLTW'  # PDRKAAVSHWQ
    antibody = 'SGSETPGTSESATPESQVQLVESGGGLVQPGGSLTLSCTASGFTLDHYDIGWFRQAPGKEREGVSCINNSDDDTYYADSVKGRFTIFMNNAKDTVYLQMNSLKPEDTAIYYCAEARGCKRGRYEYDFWGQGTQVTVSS'
    limit_range = [[42, 51], [65, 75], [112, 127]]
    find_interest(antibody, antigen, limit_range, 10, mode='convexhull')