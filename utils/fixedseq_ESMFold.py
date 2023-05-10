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
    L = len(self.x_seqs)
    vocab = residue_constants.restypes
    K = len(vocab)

    # restricted regions to mutate
    choices = []
    for i in cfg.limit_range:
        for j in range(i[0], i[1]):
            choices.append(j)

    # print the loaded protein sequence
    init_seqs = self.x_seqs
    print(f'Init sequence: {init_seqs}')
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
    for step, s_cfg in itr:
        x = self.x_seqs
        a_cfg = s_cfg.accept_reject

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
            log_P_x = self.calc_total_loss(x, s_cfg)[0]  # [B]
            self.best_seq.append([-1, log_P_x.item(), x])
            # import pdb; pdb.set_trace()

        log_P_xp = self.calc_total_loss(xp, s_cfg)[0]  # [B]
        if len(self.best_seq) < s_cfg.keep_best or log_P_xp < self.best_seq[-1][1]:
            print(step, log_P_xp.item())
            self.best_seq.append([step, log_P_xp.item(), xp])
            self.best_seq = sorted(self.best_seq, key=lambda x: x[1])[:s_cfg.keep_best]
        log_A_xp_x = (-log_P_xp - -log_P_x) / a_cfg.temperature  # [B]
        A_xp_x = (log_A_xp_x).exp().clamp(0, 1)  # [B]
        # A_xp_x = log_A_xp_x.sigmoid()  # [B]
        # import pdb; pdb.set_trace()
        A_bools = Bernoulli(A_xp_x).sample().bool()  # [B]
        self.x_seqs = xp if A_bools else x
        if A_bools:
            log_P_x = log_P_xp.clone()

        # print and save mid outputs
        if step and step % cfg.save_interval == 0:
            diff_point = count_diff(self.x_seqs, init_seqs)
            # print(f'Mid output ({step}/{cfg.num_iter}) has changed {diff_point} amino acids.')
            if not os.path.exists(os.path.dirname(cfg.path)):
                os.makedirs(os.path.dirname(cfg.path))
            if not os.path.exists(cfg.pdb_dir):
                os.makedirs(cfg.pdb_dir)
            with open(cfg.path, 'a') as f:
                f.write(f'>sample_iter{step}_loss{log_P_x.item()}\n')
                f.write(f'{self.x_seqs}\n')
                # print(f"Write to {cfg.path}: \n {self.x_seqs}")
            pdbfile = self.struct_model.output_to_pdb(self.fold_output)
            for i in range(len(pdbfile)):
                with open(cfg.pdb_dir+f'/iter_{step}_{i}.pdb', 'w') as f:
                    f.write(pdbfile[i])