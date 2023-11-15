# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import logging
import random
import queue

import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli, OneHotCategorical
from tqdm import tqdm
from openfold.np import residue_constants
from esm.data import Alphabet


logger = logging.getLogger(__name__)


def count_diff(a, b):
    assert len(a) == len(b)
    cnt = sum([1 if a[i] != b[i] else 0 for i in range(len(a))])
    return cnt


def substitute(s: str, i: int, tar: str):
    s = list(s)
    s[i] = tar
    s = "".join(s)
    return s


@torch.no_grad()
def stage_Astar_fold(self, cfg, disable_tqdm=False):
    """Astar search + multi-branch"""
    vocab = residue_constants.restypes
    K = len(vocab)
    L =  len(self.x_seqs)

    # restricted regions to mutate
    choices = []
    for i in cfg.limit_range:
        choices += list(range(*i))

    # print the loaded protein sequence
    self.queue_activity = queue.PriorityQueue()
    for seq in self.x_seqs:
        self.queue_activity.put((-1., seq))
    self.visited = dict()
    
    print(f'Init queue: {self.queue_activity.qsize()}')
    print(f'Mutation range: {cfg.limit_range} / {L}')

    self.best_activity = 0
    itr = self.stepper(range(cfg.num_iter), cfg=cfg)
    itr = tqdm(itr, total=cfg.num_iter, disable=disable_tqdm)
    heur = cfg.accept_reject.get('heuristic_evolution', False)
    if heur:
        alphabet = Alphabet.from_architecture('ESM-1b')
        
    for step, s_cfg in itr:
        x = self.queue_activity.get()
        updated = False

        if not heur:
            plans = [[p, a] for p in range(len(choices)) for a in range(K)]
            while True:
                plan = plans.pop(random.randint(0, len(plans) - 1))
                xp = substitute(x, choices[plan[0]], vocab[plan[1]])
                if xp not in self.visited:
                    break
        else:  # Heuristic selection of AA
            pos = choices[torch.randint(0, len(choices), (1,))[0]]
            batch_converter = alphabet.get_batch_converter()
            self.struct_model.esm.eval()
            data = [('protein'), substitute(x, pos, '<mask>')]
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
            with torch.no_grad():
                results = self.struct_model.esm(batch_tokens, repr_layers=[33])
                assert results['logits'][0].shape[0] == len(x) + 2  # <cls>, ..., <eos>
                AA_prob = results['logits'][0][pos+1].softmax()
        
            while True:
                target = random.choices(alphabet.all_toks, weights=AA_prob, k=1)
                xp = substitute(x, pos, target)
                if (target in vocab) and (xp not in self.visited):
                    break
                AA_prob[alphabet.all_toks.index(target)] = 0


        total_loss, logs = self.calc_total_loss(xp, s_cfg)
        confidence, activity = logs['fold_conf'], logs['activity']
        self.visited[xp] = (confidence, activity)
        
        if confidence > 0.8:
            self.queue_activity.put((activity, xp))
        
        if len(self.best_seq) < s_cfg.keep_best or log_P_xp < self.best_seq[-1][1]:
            # print(step, logs_xp)
            self.best_seq.append([step, log_P_xp.item(), xp, logs_xp])
            self.best_seq = sorted(self.best_seq, key=lambda x: x[1])[:s_cfg.keep_best]
            updated = True
        log_A_xp_x = (-log_P_xp - -log_P_x) / s_cfg.accept_reject.temperature  # [B]
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
                f.write(f'>sample_iter{step}_{logs_x}_Diff{diff_point}\n')
                f.write(f'{self.x_seqs}\n')

            if not cfg.pdb_dir: continue
            if not os.path.exists(cfg.pdb_dir):
                os.makedirs(cfg.pdb_dir)
            pdbfile = self.struct_model.output_to_pdb(self.fold_output)
            for i in range(len(pdbfile)):
                with open(cfg.pdb_dir+f'/iter_{step}_{i}.pdb', 'w') as f:
                    f.write(pdbfile[i])