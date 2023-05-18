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

from scipy.spatial import ConvexHull
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
    standard_AA = 'LAGVSERTIDPKQNFYMHWC'

    def __init__(
        self,
        cfg,
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

        # load model
        if not self.cfg.get('debug', False):
            self.struct_model = esm.pretrained.esmfold_v1().to(self.device)

        set_rng_seeds(self.seed)
        self.schedulers = {}  # reset schedulers
        self.resuming_stage = False

        assert self.cfg.antibody is not None
        self.x_seqs = self.cfg.antibody
        self.init_seqs = self.x_seqs

        torch.backends.cudnn.benchmark = True  # Slightly faster runtime for optimization
        logger.info("Finished Designer init")

    def calc_fold_loss(self, x_seq, antigen, objects, limit_range, selection, reduction, num_recycles):
        l_ag = [len(i) for i in antigen]

        output = self.struct_model.infer([ag + x_seq for ag in antigen], num_recycles=num_recycles)
        # for outside usage
        self.fold_output = output

        # average on all atoms of a protein
        idx = output["atom37_atom_exists"]
        idx_sum = idx.sum(dim=2)
        plddt = (output["plddt"] * idx).sum(dim=2) / idx_sum
        xyz = atom14_to_atom37(output["positions"][-1], output)  # [B, L, 3]
        xyz = (xyz * idx[..., None].repeat(1,1,1,3)).sum(dim=2) / idx_sum[..., None].repeat(1,1,3)

        res = self.calc_distance(xyz, antigen, limit_range, objects,
            selection=selection, reduction=reduction)

        return torch.tensor(res), plddt

    def calc_fold_conf(self, x, plddt, cfg):
        conf = []
        l_ag = [len(i) for i in cfg.antigen]
        for a in range(len(l_ag)):
            assert plddt.shape[1] == l_ag[a] + len(x)
            if cfg.get('focus_on_antigen', False):
                idx = list(range(*cfg.objects[a]))
                confidence = plddt[a, idx].max() / 100
            else:
                idx = list(range(l_ag[a] + cfg.len_linker, l_ag[a] + len(x))) + \
                    list(range(*cfg.objects[a]))
                confidence = plddt[a, idx].mean() / 100
            conf.append(confidence)
        return torch.tensor(conf)

    def calc_total_loss(self, x, cfg):
        """
            Easy one-stop-shop that calls out to all the implemented loss calculators,
            aggregates logs, and weights total_loss.
        """

        logs = {}
        total_loss = 0
        e_cfg = cfg.accept_reject.energy_cfg
        if e_cfg.dist_w:
            dist_m_nlls = self.calc_dist_loss(x, cfg.objects, e_cfg.selection, e_cfg.reduction)
            dist_m_nlls *= dist_w
            total_loss += dist_m_nlls
            logs['dist_loss'] = dist_m_nlls
        if e_cfg.fold_w:
            fold_m_nlls, plddt = self.calc_fold_loss(x, cfg.antigen, cfg.objects, cfg.limit_range,
                e_cfg.selection, e_cfg.reduction, e_cfg.num_recycles)
            fold_m_nlls = (fold_m_nlls * torch.tensor(e_cfg.fold_w)).sum()
            logs['fold_loss'] = fold_m_nlls.item()
            fold_conf = self.calc_fold_conf(x, plddt, cfg)
            logs['fold_conf'] = fold_conf.item()
            if e_cfg.conf_nonlinear == 'relu':
                fold_conf[fold_conf < 0.85] = 0
            elif e_cfg.conf_nonlinear == 'leakyrelu':
                fold_conf[fold_conf < 0.85] = fold_conf[fold_conf < 0.85] / 2
            elif e_cfg.conf_nonlinear in [2, '2']:
                fold_conf = fold_conf.pow(2)
            elif e_cfg.conf_nonlinear in [3, '3']:
                fold_conf = fold_conf.pow(3)
            fold_conf_loss = (fold_conf * torch.tensor(e_cfg.conf_w)).sum()
            logs['fold_conf_loss'] = fold_conf_loss.item()
            total_loss += fold_m_nlls + fold_conf_loss
        logs['total_loss'] = total_loss.item()

        return total_loss, logs  # [B], Dict[str:[B]]

    def calc_distance(self, xyz, antigen, limit_range, objects, selection='min', reduction='mean'):
        if isinstance(antigen, str):
            antigen = [antigen]
        l_ag = [len(i) for i in antigen]
        res = []
        for a in range(xyz.shape[0]):
            dist = []
            for t in range(len(limit_range)):
                dist_hand = []
                for j in range(*limit_range[t]):
                    dist_ob = [
                        torch.norm(xyz[a, i] - xyz[a, j + l_ag[a]], p = 2)
                        for i in range(*objects[a])
                    ]
                    dist_hand.append(torch.tensor(dist_ob).mean())
                dist_hand = torch.tensor(dist_hand)
                if selection == 'min':
                    dist.append(dist_hand.min())
                elif selection == 'max':
                    dist.append(dist_hand.max())
                else:
                    dist.append(dist_hand.mean())

            if reduction == 'mean':
                dist = torch.tensor(dist).mean()
            elif reduction == 'prod':
                dist = torch.tensor(dist).prod()
            else:
                raise NotImplementedError(f'No such reduction: {reduction}')
            res.append(dist)
        return res

    def find_interest(self, 
        antigen, 
        antibody, 
        len_insterest, 
        limit_range, 
        num_recycles=4, 
        mode='brute'
    ):
        if isinstance(len_insterest, int):
            len_insterest = [len_insterest]

        self.struct_model.set_chunk_size(128)
        output = self.struct_model.infer([antigen + antibody], num_recycles=num_recycles)
        # average on all atoms of a protein
        idx = output["atom37_atom_exists"]
        idx_sum = idx.sum(dim=2)
        plddt = (output["plddt"] * idx).sum(dim=2) / idx_sum
        xyz = atom14_to_atom37(output["positions"][-1], output)  # [B, L, 3]
        xyz = (xyz * idx[..., None].repeat(1,1,1,3)).sum(dim=2) / idx_sum[..., None].repeat(1,1,3)

        losses = []
        if mode == 'brute':
            for l in len_insterest:
                for i in range(len(antigen) - l):
                    losses.append([
                        self.calc_distance(xyz, antigen, limit_range, [[i, i+l]]), i, i+l])
        elif mode == 'convexhull':
            hull = ConvexHull(xyz[0].cpu().numpy())
            for l in len_insterest:
                for i in range(len(antigen) - l):
                    if sum([i in hull.vertices for i in range(i,i+l)]) == 0:
                        continue
                    losses.append([
                        self.calc_distance(xyz, antigen, limit_range, [[i, i+l]]), i, i+l])
        
        losses = sorted(losses, key=lambda x: x[0])
        return losses

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
        self.is_scheduler_registered = True
    
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
        if not hasattr(self, 'is_scheduler_registered'):
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