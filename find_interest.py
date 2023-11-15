import os
import hydra
from tqdm import tqdm
from lm_design_fold import Designer

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial import ConvexHull
from utils.icp import calc_struct_sim


def search_interest(antibody, antigen, limit_range, len_insterest, mode='brute'):
    if isinstance(len_insterest, int):
        len_insterest = [len_insterest]
        
    TASK = "fixedseqs_fold"
    num_recycles = 4
    conf_w = [0]
    struct_w = [1]
    # Load hydra config from config.yaml
    with hydra.initialize_config_module(config_module="conf"):
        cfg = hydra.compose(
            config_name="config_fold_3fingers",
            overrides=[
                f"task={TASK}",
                "debug=False",
                f'tasks.fixedseqs_fold.accept_reject.energy_cfg.conf_w={conf_w}',
                f'tasks.fixedseqs_fold.accept_reject.energy_cfg.num_recycles={num_recycles}',
            ])

    # Create a designer from configuration
    des = Designer(cfg)

    # Calculate the antigen's structure
    des.struct_model.set_chunk_size(128)
    output = des.struct_model.infer([antigen], num_recycles=num_recycles)
    xyz_antigen, plddt = des.get_xyz(output)
    print(f'[LOG]  Obtain antigen\'s structure.')
    xyz_antigen = xyz_antigen[0].cpu().numpy()

    # Find candidates on the antigen
    candidates = []
    if mode == 'brute':
        for l in len_insterest:
            for i in range(len(antigen) - l):
                candidates.append([i, l])
    elif mode == 'convexhull':
        hull = ConvexHull(xyz_antigen)
        pos_dict = {i:i in hull.vertices for i in range(len(antigen))}
        for l in len_insterest:
            for i in range(len(antigen) - l):
                if sum([pos_dict[i] for i in range(i, i+l)]) == 0:
                    continue
                candidates.append([i, l])
    print(f'[LOG]  Obtain {len(candidates)} candidates using {mode}.')
    # import pdb; pdb.set_trace()
    # calculate loss for each candidate
    losses = []
    for i, l in tqdm(candidates):
        des.cfg.tasks[cfg.task].antigen = [antigen[i:i+l]]
        des.cfg.tasks[cfg.task].limit_range = [[0, l]]
        loss, logs = des.calc_total_loss(antibody, cfg.tasks[cfg.task])
        
        # for structural similarity
        fitness, rmse = calc_struct_sim(xyz_antigen[i:i+l], 
            des.xyz[0, 0:l].cpu().numpy(), cuda=False)
        struct_loss = rmse * struct_w[0]
        loss += struct_loss
        logs['struct_loss'] = struct_loss
        logs['struct_fitness'] = fitness

        losses.append([loss, logs, i, i+l])

    # Select the best candidate with the least loss
    losses = sorted(losses, key=lambda x: x[0])
    print(losses)
    np.save(f'output/interest/losses_conf{conf_w[0]}_struct{struct_w[0]}.npy', losses)
    print(antigen[losses[0][2]:losses[0][3]])


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
    np.save
    print(losses)


if __name__ == '__main__':
    antigen = 'WRQTWSGPGPDRKAAVSHWQQVQRDMFTLEDTLLGYLADDLTW'  # PDRKAAVSHWQ
    antibody = 'SGSETPGTSESATPESQVQLVESGGGLVQPGGSLTLSCTASGFTLDHYDIGWFRQAPGKEREGVSCINNSDDDTYYADSVKGRFTIFMNNAKDTVYLQMNSLKPEDTAIYYCAEARGCKRGRYEYDFWGQGTQVTVSS'
    limit_range = [[42, 51], [65, 75], [112, 127]]
    # find_interest(antibody, antigen, limit_range, 10, mode='convexhull')
    search_interest(antibody, antigen, limit_range, 11, mode='brute')