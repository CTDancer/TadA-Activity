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


logger = logging.getLogger(__name__)


def count_diff(a, b):
    assert len(a) == len(b)
    cnt = sum([1 if a[i] != b[i] else 0 for i in range(len(a))])
    return cnt


@torch.no_grad()
def stage_fixedseqs(self, cfg, disable_tqdm=False):
    """Metropolis-Hastings sampling with uniform proposal and energy-based acceptance."""
    B, L, K = self.x_seqs.shape

    # restricted regions to mutate
    choices = []
    for i in cfg.limit_range:
        for j in range(i[0], i[1]):
            choices.append(j)

    # print the loaded protein sequence
    init_seqs = self.decode(self.x_seqs)[0]
    print(f'Init sequence: {init_seqs}')
    print(f'Mutation range: {cfg.limit_range} / {L}')
    for i in cfg.limit_range:
        print(f"Mutation strings: {init_seqs[i[0]:i[1]]}")

    # uniform proposal distribution.
    log_p_x_i = torch.full((B, K), fill_value=-float("inf")).to(self.x_seqs)  # [B, K]
    log_p_x_i[..., self.vocab_mask_AA] = 0  # [B, K]
    p_x_i = log_p_x_i.softmax(-1)

    # random init for the given range
    if cfg.limit_range_random_init:
        for c in choices:
            init_i = OneHotCategorical(probs=p_x_i).sample()
            self.x_seqs[0, c] = init_i
        print(f'Random init sequence: {self.decode(self.x_seqs)[0]}')
    
    self.best_seq = []
    itr = self.stepper(range(cfg.num_iter), cfg=cfg)
    itr = tqdm(itr, total=cfg.num_iter, disable=disable_tqdm)
    for step, s_cfg in itr:
        x = self.x_seqs
        a_cfg = s_cfg.accept_reject

        ##############################
        # Proposal
        ##############################
        # Decide which position to mutate == {i}.
        # mask 1 place
        mask = torch.zeros((B, L, 1), dtype=torch.bool).to(x)  # [B,L,1]
        idx = torch.randint(0, len(choices), (B,))
        mask[:, choices[idx]] = True  # [1,L,1]
        mask = mask.bool()

        xp_i = OneHotCategorical(probs=p_x_i).sample()
        xp = x.masked_scatter(mask, xp_i)  # [B,L,K]

        ##############################
        # Accept / reject
        ##############################
        # log A(x',x) = log P(x') - log P(x))
        # for current input x, proposal x', target distribution P and symmetric proposal.
        if not self.best_seq:
            log_P_x = self.calc_total_loss(x, mask, **a_cfg.energy_cfg)[0]  # [B]
            self.best_seq.append([-1, log_P_x.item(), x])
            # import pdb; pdb.set_trace()

        log_P_xp = self.calc_total_loss(xp, mask, **a_cfg.energy_cfg)[0]  # [B]
        if len(self.best_seq) < 5 or log_P_xp < self.best_seq[-1][1]:
            print(step, log_P_xp.item())
            self.best_seq.append([step, log_P_xp.item(), xp])
            self.best_seq = sorted(self.best_seq, key=lambda x: x[1])[:5]
        log_A_xp_x = (-log_P_xp - -log_P_x) / a_cfg.temperature  # [B]
        A_xp_x = (log_A_xp_x).exp().clamp(0, 1)  # [B]
        # A_xp_x = log_A_xp_x.sigmoid()  # [B]
        A_bools = Bernoulli(A_xp_x).sample().bool()  # [B]
        self.x_seqs = torch.where(A_bools[:, None, None], xp, x)  # [B,L,K]
        if A_bools[0]:
            log_P_x = log_P_xp.clone()

        # print and save mid outputs
        if step and step % cfg.save_interval == 0:
            diff_point = count_diff(self.x_seqs[0].argmax(-1), self.wt_seq[0].argmax(-1))
            # print(f'Mid output ({step}/{cfg.num_iter}) has changed {diff_point} amino acids.')
            if not os.path.exists(os.path.dirname(cfg.path)):
                os.makedirs(os.path.dirname(cfg.path))
            with open(cfg.path, 'a') as f:
                mid_seq = self.decode(self.x_seqs)[0]
                f.write(f'>sample_iter{step}\n')
                f.write(f'{mid_seq}\n')
                # print(f"Write to {cfg.path}: \n {mid_seq}")