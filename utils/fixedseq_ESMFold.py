# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import logging

import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli, OneHotCategorical
from tqdm import tqdm
from openfold.np import residue_constants

logger = logging.getLogger(__name__)


def count_diff(a, b):
    assert len(a) == len(b)
    cnt = sum([1 if a[i] != b[i] else 0 for i in range(len(a))])
    return cnt


def substitute(s: str, i: int, target: str):
    s = list(s)
    s[i] = target
    s = "".join(s)
    return s


@torch.no_grad()
def stage_fixedseqs_fold(self, cfg, disable_tqdm=False):
    """Metropolis-Hastings sampling with uniform proposal and energy-based acceptance."""
    B = 1
    vocab = residue_constants.restypes
    K = len(vocab)
    inv = cfg.get('inv', False)
    L = len(cfg.antigen[0]) if inv else len(self.x_seqs)

    # restricted regions to mutate
    if inv:
        choices = list(range(*cfg.objects[0]))
    else:
        choices = []
        for i in cfg.limit_range:
            for j in range(i[0], i[1]):
                choices.append(j)

    # print the loaded protein sequence
    self.x_seqs = cfg.antigen[0] if inv else self.x_seqs
    init_seqs = self.x_seqs
    print(f'Init sequence: {init_seqs}')
    if not inv:
        print(f'Mutation range: {cfg.limit_range} / {L}')
        for i in cfg.limit_range:
            print(f"Mutation strings: {init_seqs[i[0]:i[1]]}")

    # random init for the given range
    if cfg.limit_range_random_init:
        for c in choices:
            init_i = torch.randint(0, K, (B, ))
            self.x_seqs = substitute(self.x_seqs, c, vocab[init_i[0]])
        print(f'Random init sequence: {self.x_seqs}')
    
    self.best_seq = []
    itr = self.stepper(range(cfg.num_iter), cfg=cfg)
    itr = tqdm(itr, total=cfg.num_iter, disable=disable_tqdm)
    antibody = self.init_seqs
    for step, s_cfg in itr:
        x = self.x_seqs
        a_cfg = s_cfg.accept_reject
        updated = False

        ##############################
        # Proposal
        ##############################
        # Decide which position to mutate == {i}. Mask 1 place
        idx = torch.randint(0, len(choices), (B,))
        target = torch.randint(0, K, (B,))
        while x[choices[idx[0]]] == vocab[target[0]]:
            target = torch.randint(0, K, (B,))
        xp = substitute(x, choices[idx[0]], vocab[target[0]])

        ##############################
        # Accept / reject
        ##############################
        # log A(x',x) = log P(x') - log P(x))
        # for current input x, proposal x', target distribution P and symmetric proposal.
        if not self.best_seq:
            if inv:
                s_cfg.antigen = [x]
                log_P_x, logs_x = self.calc_total_loss(antibody, s_cfg)  # [B]
            else:
                log_P_x, logs_x = self.calc_total_loss(x, s_cfg)  # [B]
            self.origin_seq = [-1, log_P_x.item(), x, logs_x]
            # import pdb; pdb.set_trace()

        if inv:
            s_cfg.antigen = [xp]
            log_P_xp, logs_xp = self.calc_total_loss(antibody, s_cfg)  # [B]
        else:
            log_P_xp, logs_xp = self.calc_total_loss(xp, s_cfg)  # [B]
        
        if len(self.best_seq) < s_cfg.keep_best or log_P_xp < self.best_seq[-1][1]:
            # print(step, logs_xp)
            self.best_seq.append([step, log_P_xp.item(), xp, logs_xp])
            self.best_seq = sorted(self.best_seq, key=lambda x: x[1])[:s_cfg.keep_best]
            updated = True
        log_A_xp_x = (-log_P_xp - -log_P_x) / a_cfg.temperature  # [B]
        A_xp_x = (log_A_xp_x).exp().clamp(0, 1)  # [B]
        # A_xp_x = log_A_xp_x.sigmoid()  # [B]
        # import pdb; pdb.set_trace()
        A_bools = Bernoulli(A_xp_x).sample().bool()  # [B]
        self.x_seqs = xp if A_bools else x
        if A_bools:
            log_P_x = log_P_xp.clone()
            logs_x = logs_xp

        # show and save mid outputs
        if cfg.save_interval == 'best':
            flag_save = updated
        else:
            flag_save = (step and (step % cfg.save_interval == 0))

        if flag_save:
            diff_point = count_diff(self.x_seqs, init_seqs)
            # print(f'Mid output ({step}/{cfg.num_iter}) has changed {diff_point} amino acids.')
            if not os.path.exists(os.path.dirname(cfg.path)):
                os.makedirs(os.path.dirname(cfg.path))
            with open(cfg.path, 'a') as f:
                f.write(f'>sample_iter{step}_{logs_x}\n')
                f.write(f'{self.x_seqs}\n')

            if not cfg.pdb_dir: continue
            if not os.path.exists(cfg.pdb_dir):
                os.makedirs(cfg.pdb_dir)
            pdbfile = self.struct_model.output_to_pdb(self.fold_output)
            for i in range(len(pdbfile)):
                with open(cfg.pdb_dir+f'/iter_{step}_{i}.pdb', 'w') as f:
                    f.write(pdbfile[i])