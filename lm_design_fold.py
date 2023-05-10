# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# core
import logging
import os
import sys
import time

from omegaconf import DictConfig
import hydra
import os
from pathlib import Path
import sys
import time
import logging

import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn.functional as F

# make sure script started from the root of the this file
assert Path.cwd().name == 'Interact'
import esm
from esm.data import Alphabet

from utils.scheduler import SchedulerSpec, to_scheduler, set_scheduler_repo
import utils.pdb_loader as pdb_loader
from utils.loss import get_cce_loss
from utils.lm import lm_marginal
from utils.masking import assert_valid_mask
from utils.sampling import (
    set_rng_seeds,
)
from utils.constants import COORDS_ANGLE_NAMES, COORDS4D_NAMES
import utils.struct_models as struct_models
from utils.free_generation import stage_free_generation
from utils.fixedbb import stage_fixedbb
from utils.fixedseq import stage_fixedseqs
from utils.fixedseq_ESMFold import stage_fixedseqs_fold
from utils.lm import WrapLmEsm

from utils.tensor import (
    assert_shape,
)
from utils import ngram as ngram_utils
from openfold.utils.feats import atom14_to_atom37


logger = logging.getLogger(__name__)  # Hydra configured
os.environ['MKL_THREADING_LAYER'] = 'GNU'


class Designer:
    cutoff_dist = 8
    LOGITS_LARGE = 100
    standard_AA = 'LAGVSERTIDPKQNFYMHWC'

    ##########################################
    # Inits
    ##########################################
    def __init__(
        self,
        cfg,
        seq,
        device=None,
        fp16=False
    ):
        self.fp16 = fp16
        ## Initialize models
        if device:
            self.device = device
        else:
            use_cuda = torch.cuda.is_available() and not cfg.disable_cuda
            device_idx = f":{cfg.cuda_device_idx}" if cfg.get('cuda_device_idx') else ""
            self.device = torch.device(f'cuda{device_idx}' if use_cuda else 'cpu')
        SEED_SENTINEL = 1238
        self.seed = cfg.seed + SEED_SENTINEL
        self.cfg = cfg
        self.seq = seq
        self.allowed_AA = ''.join(AA for AA in self.standard_AA if (
                ('suppress_AA' not in self.cfg) or (not AA in self.cfg.suppress_AA)))

        self._init_models()

        set_rng_seeds(self.seed)
        self.schedulers = {}  # reset schedulers
        self.resuming_stage = False
        self.init_sequences(cfg.num_seqs)

        torch.backends.cudnn.benchmark = True  # Slightly faster runtime for optimization
        logger.info("Finished Designer init")

    def _init_models(self):
        self.vocab = Alphabet.from_architecture('ESM-1b')
        self.vocab_mask_AA = torch.BoolTensor(
            [t in self.allowed_AA for t in self.vocab.all_toks]
        ).to(self.device)
        self.vocab_mask_AA_idx = torch.nonzero(self.vocab_mask_AA).squeeze()

        # for esmfold
        if self.cfg:
            self.struct_model = esm.pretrained.esmfold_v1().to(self.device)
            return
            
        self.struct_model, self.pdb_loader_params = struct_models.load(
            self.vocab,
        )
        if self.fp16:
            self.struct_model.lm = self.struct_model.lm.half()
        self.LM = WrapLmEsm(self.struct_model.lm, self.vocab)

        # 4. Common model settings
        def apply_common_settings(model):
            model.to(self.device)
            model.eval()
            # No grads for models
            for p in model.parameters():
                p.requires_grad = False
            return model

        self.LM = apply_common_settings(self.LM)
        self.struct_model = apply_common_settings(self.struct_model)

    def init_sequences(self, num_seqs):
        if self.seq is not None:
            self.x_seqs = self.seq
            self.init_seqs = self.x_seqs
            return

        assert num_seqs == 1, "Only 1 sequence design in parallel supported for now."
        self.B = B = self.num_seqs = num_seqs
        
        if self.cfg.get('seq_init_random', True):
            K = len(self.vocab)
            AA_indices = torch.arange(K, device=self.device)[self.vocab_mask_AA]
            bt = torch.from_numpy(np.random.choice(AA_indices.cpu().numpy(), size=(B, self.L))).to(self.device)
            self.x_seqs = F.one_hot(bt,K).float() if not self.fp16 else F.one_hot(bt,K).to(torch.float16)
        else:
            self.x_seqs = self.wt_seq
            self.init_seqs = self.x_seqs.clone()

    ##########################################
    # Losses
    ##########################################
    
    def calc_dist_loss(self, x_seq, objects, selection, reduction, 
        discrete=True, w_conf=1.0):
        '''
            objects: [[range1, range2], ...], calculate the minimum atom distance between range1 and range2
                e.g., [[[0,7], [13,17]], ...]
        '''
        B, L, K = x_seq.shape
        res_preds = self.struct_model(x_seq)

        # bin to class weight (linear since "np.linspace" in pdb_loader)
        n_bin = res_preds['p_dist'].shape[-1]
        bin_weight = torch.tensor([i for i in range(n_bin)]).exp().to(res_preds['p_dist'])
        bound = torch.tensor(n_bin - 1).exp()

        def calc_dist(logits):
            if discrete:
                conf, res = logits.max(dim=0)
                res = res - conf * w_conf
            else:
                res = logits.dot(bin_weight) / bound
            return res

        dist = []
        for ob in objects:
            dist_ob = []
            for j in range(*ob[1]):
                dist_hand = []
                for i in range(*ob[0]):
                    dist_hand.append(calc_dist(res_preds['p_dist'][0, i, j]))
                dist_ob.append(torch.tensor(dist_hand).mean())
            dist_ob = torch.tensor(dist_ob)
            if selection == 'min':
                dist.append(dist_ob.min())
            elif selection == 'max':
                dist.append(dist_ob.max())
            else:
                dist.append(dist_ob.mean())
        # import pdb; pdb.set_trace()
        if reduction == 'mean':
            return torch.tensor(dist).mean()
        elif reduction == 'prod':
            return torch.tensor(dist).prod()
        else:
            raise NotImplementedError(f'No such reduction: {reduction}')

    def calc_fold_loss(self, x_seq, objects, selection, reduction, w_conf=1.0):
        output = self.struct_model.infer(x_seq, num_recycles=1)

        # average on all atoms of a protein
        plddt = output["plddt"].mean(dim=2)
        xyz = atom14_to_atom37(output["positions"][-1], output).mean(dim=2)  # [L, 3]

        def calc_dist(x1, x2):
            res = torch.norm(x1 - x2, p=2)
            return res

        dist = []
        for ob in objects:
            dist_ob = []
            for j in range(*ob[1]):
                dist_hand = []
                for i in range(*ob[0]):
                    dij = calc_dist(xyz[0, i], xyz[0, j])
                    dist_hand.append(dij)
                dist_ob.append(torch.tensor(dist_hand).mean())
            dist_ob = torch.tensor(dist_ob)

            if selection == 'min':
                dist.append(dist_ob.min())
            elif selection == 'max':
                dist.append(dist_ob.max())
            else:
                dist.append(dist_ob.mean())

        # import pdb; pdb.set_trace()
        if reduction == 'mean':
            res = torch.tensor(dist).mean()
        elif reduction == 'prod':
            res = torch.tensor(dist).prod()
        else:
            raise NotImplementedError(f'No such reduction: {reduction}')
        
        if w_conf > 0:
            care_idx = [list(range(*j)) for i in objects for j in i]
            idx = list(set([j for i in care_idx for j in i]))
            confidence = plddt[0, idx].mean().to(res)
            res += w_conf * confidence / 100
        
        return res

    def calc_total_loss(
        self, 
        x,
        LM_w, 
        struct_w,
        ngram_w, ngram_orders,
        dist_w, objects, selection, reduction,
        fold_w,
        temp_struct=None):
        """
        Easy one-stop-shop that calls out to all the implemented loss calculators,
        aggregates logs, and weights total_loss.

        As a refresher:
            calc_sequence_loss:
                calculates \sum log p(x_i|x_\i) for i in {set bits in mask}.
                    If mask is all ones, this is equal to Pseudo-log-likelihood.
                NOTE: every position in mask is masked *separately*
                    Therefore, there will be multiple forward passes of the LM.
            calc_structure_loss:
                calculates p(y|x)
            calc_ngram_loss:
                calculates p_ngram(x)
        """

        logs = {}
        total_loss = 0
        if LM_w:
            lm_m_nlls, _, lm_loss_dict = self.calc_sequence_loss(x, mask=mask)
            lm_m_nlls *= LM_w / self.L 
            total_loss += lm_m_nlls
            logs['lm_loss'] = lm_m_nlls
            logs.update(lm_loss_dict)
        if struct_w:
            struct_m_nlls, struct_loss_dict = self.calc_structure_loss(x, temp_struct=temp_struct)
            struct_m_nlls *= struct_w
            total_loss += struct_m_nlls
            logs['struct_loss'] = struct_m_nlls
            logs.update(struct_loss_dict)
        if ngram_w:
            ngram_m_nlls = self.calc_ngram_loss(x, ngram_orders=ngram_orders)
            ngram_m_nlls *= ngram_w
            total_loss += ngram_m_nlls
            logs['ngram_loss'] = ngram_m_nlls
        if dist_w:
            dist_m_nlls = self.calc_dist_loss(x, objects, selection, reduction)
            dist_m_nlls *= dist_w
            total_loss += dist_m_nlls
            logs['dist_loss'] = dist_m_nlls
        if fold_w:
            fold_m_nlls = self.calc_fold_loss(x, objects, selection, reduction)
            fold_m_nlls *= fold_w
            total_loss += fold_m_nlls
            logs['fold_loss'] = fold_m_nlls

        return total_loss, logs  # [B], Dict[str:[B]]


    ##########################################
    # YAML Execution
    ##########################################

    def run_from_cfg(self):
        """
        Main run-loop for the Designer. Runs a relevant design procedure from the config.
        """
        logger.info(f'Designing sequence for task: {self.cfg.task}')
        
        design_cfg = self.cfg.tasks[self.cfg.task]
        if self.cfg.task == 'fixedseq':
            stage_fixedseq(self, design_cfg)
        elif self.cfg.task == 'fixedseqs':
            stage_fixedseqs(self, design_cfg)
        elif self.cfg.task == 'fixedseqs_fold':
            stage_fixedseqs_fold(self, design_cfg)
        else:
            raise ValueError(f'Invalid task: {self.cfg.task}')

        self.output_seq = self.x_seqs
            
    def init_schedulers_from_cfg(self, cfg: DictConfig):
        """
        Similar to init_schedulers, but expects a stage-specific DictConfig.
        Populates self.schedulers with dotlist key.
        (Simplifies later OmegaConf accesses)
        Example:
        cfg = {
            num_iter: 10,
            sub_cfg: {
                my_sched: {
                    scheduler: CosineAnnealingLR
                    initial: 1e-2
                    T_max: 200}}}
        Effect:
            self.schedulers['sub_cfg.my_sched'] = <Scheduler>
        """
        def walk_cfg(d, parent_key='', sep='.'):
            from collections.abc import MutableMapping
            for k, v in d.items():
                new_key = parent_key + sep + k if parent_key else k
                yield (new_key, v)
                if isinstance(v, MutableMapping):
                    yield from walk_cfg(v, new_key, sep=sep)
                    
        from typing import Optional, Dict, List, Any, Union
        def is_sspec(maybe_sspec: Union[SchedulerSpec, Any]):
            infer_from_key = (isinstance(maybe_sspec, DictConfig) 
                and maybe_sspec.get('scheduler', None) is not None)
            # infer_from_type = OmegaConf.get_type(maybe_sspec) is SchedulerSpec
            return infer_from_key

        if not self.resuming_stage:
            for name, maybe_sspec in walk_cfg(cfg, sep='.'):
                if is_sspec(maybe_sspec):
                    assert not name in self.schedulers, f"Trying to re-register {name}"
                    self.schedulers[name] = to_scheduler(maybe_sspec)
    
    def gen_step_cfg(self, cfg):
        """
        Replace schedulers in a cfg with step-specific values.
        Make sure to call `init_schedulers_from_cfg(cfg)` first!
        Uses Designer state:
            - self.schedulers
        """
        step_cfg = cfg.copy()
        for name, sched in self.schedulers.items():
            if OmegaConf.select(step_cfg, name) is not None:
                OmegaConf.update(step_cfg, name, sched(), merge=False)
        return step_cfg

    def stepper(self, iterable, update_schedulers=True, cfg=None):
        self.init_schedulers_from_cfg(cfg)
        
        for local_step in iterable:
            yield local_step, self.gen_step_cfg(cfg)

            if update_schedulers:
                self.update_schedulers()

    def update_schedulers(self):
        for s in self.schedulers.values():
            try:
                s.step()
            except AttributeError:
                pass  # constants: dummy lambda

    def init_schedulers(self, **kwargs):
        """
        Schedulers (always stage-specific) are initialized according to SchedulerSpec,
        and depend on global_step
        Optionally wrapping an optimizer class with single param group.
        Stores the schedulers in self._schedulers
        Returns:
            functions which return the current value for each
        """
        set_scheduler_repo(self.cfg.get('schedulers', {}))
        for name, sspec in kwargs.items():
            assert not name in self.schedulers, f"Trying to re-register {name}"
            self.schedulers[name] = to_scheduler(sspec)
        assert sys.version_info >= (3, 6), "py>=3.6 preserve kwarg and dict order see PEP468"
        return [self.schedulers[name] for name in kwargs]


@hydra.main(config_path="conf/", config_name="config")
def main(cfg: DictConfig) -> None:
    args_no_spaces = [arg.replace(" ", "") for arg in sys.argv[1:]]
    logger.info(f"Running with args: {' '.join(args_no_spaces)}")

    pdb_fn = cfg.pdb_fn
    logger.info(f'Starting to optimize seq for {pdb_fn}')

    start_time = time.time()
    des = Designer(cfg, pdb_fn)

    des.run_from_cfg()    
    logger.info("finished after %s hours", (time.time() - start_time) / 3600)


if __name__ == "__main__":
    main()  # noqa pylint: disable=no-value-for-parameter
