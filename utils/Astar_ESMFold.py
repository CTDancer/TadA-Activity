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


def substitute(s: str, i: int, tar: str):
    s = list(s)
    s[i] = tar
    s = "".join(s)
    return s

def resume_module(state_dict):
    queue_activity = queue.PriorityQueue()  # (-act, seq, conf)
    with open(os.path.join(state_dict, 'queue_sequences.fasta'), 'r') as f:
        lines = f.readlines()
        for l in range(0, len(lines), 2):
            act, conf = lines[l].strip().split('_')
            act = float(act.split('-')[1])
            conf = float(conf.split('-')[1])
            queue_activity.put((- act, lines[l + 1].strip(), conf))

    visited = set()
    with open(os.path.join(state_dict, 'sequences.fasta'), 'r') as f:
        lines = f.readlines()
        for l in range(0, len(lines), 2):
            visited.add(lines[l + 1].strip())
            
    best_seqs = queue.PriorityQueue()  # (act, seq, conf)
    with open(os.path.join(state_dict, 'best_sequences.fasta'), 'r') as f:
        lines = f.readlines()
        for l in range(0, len(lines), 2):
            act, conf = lines[l].strip().split('_')
            act = float(act.split('-')[1])
            conf = float(conf.split('-')[1])
            best_seqs.put((act, lines[l + 1].strip(), conf))
    return queue_activity, visited, best_seqs


@torch.no_grad()
def stage_Astar_fold(self, cfg, disable_tqdm=False):
    """Astar search"""
    vocab = residue_constants.restypes
    K = len(vocab)

    # restricted regions to mutate
    choices = []
    for i in cfg.limit_range:
        choices += list(range(*i))

    # print the loaded protein sequence
    self.queue_activity = queue.PriorityQueue()  # (-act, seq, conf)
    self.best_seqs = queue.PriorityQueue()  # (act, seq, conf)
    self.visited = set()
    self.best_activity = 0
    
    # load from a queue
    if isinstance(self.x_seqs, str):
        self.queue_activity, self.visited, self.best_seqs = resume_module(self.x_seqs)
    else:
        for seq in self.x_seqs:
            self.queue_activity.put((-1.0, seq, 0.9))
    
    print(f'Init queue: {self.queue_activity.qsize()}')
    print(f'Init visited: {len(self.visited)}')
    print(f'Init best: {self.best_seqs.qsize()}')
    print(f'Mutation range: {cfg.limit_range}')

    itr = self.stepper(range(cfg.num_iter), cfg=cfg)
    itr = tqdm(itr, total=cfg.num_iter, disable=disable_tqdm)
    heur = cfg.accept_reject.get('heuristic_evolution', False)
    if heur:
        alphabet = Alphabet.from_architecture('ESM-1b')
        
    import pdb
    # pdb.set_trace()
        
    for step, s_cfg in itr:
        x_act, x, x_conf = self.queue_activity.get()
        self.queue_activity.put((x_act * s_cfg.decline_rate, x, x_conf))

        if not heur:
            plans = [[p, a] for p in range(len(choices)) for a in range(K)]
            while True:
                plan = plans.pop(random.randint(0, len(plans) - 1))
                xp = substitute(x, choices[plan[0]], vocab[plan[1]])
                if xp not in self.visited:
                    break
        else:  # Heuristic selection of AA
            pos_prob = [1 / len(choices)] * len(choices)
            flag_new_seq = False
            while not flag_new_seq:
                pos = random.choices(choices, weights=pos_prob, k=1)
                batch_converter = alphabet.get_batch_converter()
                self.struct_model.esm.eval()
                data = [('protein'), substitute(x, pos, '<mask>')]
                batch_labels, batch_strs, batch_tokens = batch_converter(data)
                batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
                with torch.no_grad():
                    results = self.struct_model.esm(batch_tokens, repr_layers=[33])
                    assert results['logits'][0].shape[0] == len(x) + 2  # <cls>, ..., <eos>
                    AA_prob = results['logits'][0][pos+1].softmax()
            
                for _ in range(len(alphabet.all_toks)):
                    target = random.choices(alphabet.all_toks, weights=AA_prob, k=1)
                    xp = substitute(x, pos, target)
                    if (target in vocab) and (xp not in self.visited):
                        flag_new_seq = True
                        break
                    AA_prob[alphabet.all_toks.index(target)] = 0
                pos_prob[choices.index(pos)] = 0

        total_loss, logs = self.calc_total_loss(xp, s_cfg)
        activity, confidence = logs['activity'], logs['fold_conf']
        self.visited.add(xp)
        
        if confidence >= s_cfg.conf_threshold:
            self.queue_activity.put((-activity, xp, confidence))
            
            if len(self.queue_activity.queue) > 2 * s_cfg.queue_size:
                queue_tmp = queue.PriorityQueue()  # (-act, seq, conf)
                for _ in range(s_cfg.queue_size):
                    queue_tmp.put(self.queue_activity.get())
                del self.queue_activity
                self.queue_activity = queue_tmp
                

        if len(self.best_seqs.queue) < s_cfg.keep_best or activity > self.best_seqs.queue[0][0]:
            self.best_seqs.put((activity, xp, confidence))
            while len(self.best_seqs.queue) > s_cfg.keep_best:
                self.best_seqs.get()
            if not os.path.exists(os.path.dirname(cfg.best_path)):
                os.makedirs(os.path.dirname(cfg.best_path))
            with open(cfg.best_path, 'w') as f:
                for act, seq, conf in sorted(self.best_seqs.queue, key = lambda x: -x[0]):
                    f.write(f'>act-{act}_conf-{conf}\n')
                    f.write(f'{seq}\n')
            with open(cfg.queue_path, 'w') as f:
                for act, seq, conf in sorted(self.queue_activity.queue[:100]):
                    f.write(f'>act-{-act}_conf-{conf}\n')
                    f.write(f'{seq}\n')

        # show and save mid outputs
        flag_save = (step % cfg.save_interval == 0)

        if flag_save:
            if not os.path.exists(os.path.dirname(cfg.path)):
                os.makedirs(os.path.dirname(cfg.path))
            with open(cfg.path, 'a') as f:
                f.write(f'>sample_iter{step}_act-{activity}_conf-{confidence}\n')
                f.write(f'{xp}\n')