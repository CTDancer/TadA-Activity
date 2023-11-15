import logging
import os
import sys
import time
import hydra
from pathlib import Path
from scipy.spatial import ConvexHull
from omegaconf import DictConfig, OmegaConf
import torch
import esm
from torchdrug import core, data

from utils.scheduler import SchedulerSpec, to_scheduler, set_scheduler_repo
from utils.fold import get_xyz, calc_fold_conf, calc_distance
from utils.sampling import set_rng_seeds
from utils.fixedseq_ESMFold import stage_fixedseqs_fold
from utils.Astar_ESMFold import stage_Astar_fold
from utils.gearnet import bio_load_pdb, load_config


# make sure script started from the root of the this file
assert Path.cwd().name == 'Interact'

logger = logging.getLogger(__name__)  # Hydra configured
os.environ['MKL_THREADING_LAYER'] = 'GNU'


class Designer:

    def __init__(self, cfg, device=None):
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
        if self.cfg.get('regressor_cfg_path', None):
            self.regressor_init(self.cfg.regressor_cfg_path)
        
        logger.info("Finished Designer init")


    def calc_fold_loss(self, x_seq, antigen, objects, limit_range, selection, reduction, num_recycles):
        l_ag = [len(i) for i in antigen]

        output = self.struct_model.infer([ag + x_seq for ag in antigen], num_recycles=num_recycles)
        xyz, plddt = get_xyz(output)

        # for outside usage
        self.fold_output = output
        self.xyz = xyz
        
        # For single protein evolution
        if sum(l_ag) == 0:
            return 0, plddt
        
        res = calc_distance(xyz, antigen, limit_range, objects,
            selection=selection, reduction=reduction)

        return torch.tensor(res), plddt

    def regressor_init(self, regressor_cfg_path):

        # init
        cfg = load_config(regressor_cfg_path)
        dataset = core.Configurable.load_config_dict(cfg.dataset)
        task = core.Configurable.load_config_dict(cfg.task)
        task.preprocess(dataset, None, None)
        self.transform = core.Configurable.load_config_dict(cfg.transform)
        pretrained_dict = torch.load(cfg.checkpoint, map_location=torch.device('cpu'))['model']
        model_dict = task.state_dict()
        task.load_state_dict(pretrained_dict)
        self.task = task.cuda(self.device)
        self.task.eval()

    def calc_regressor(self,):
        pdbfile = self.struct_model.output_to_pdb(self.fold_output)
        if not os.path.exists('output/tmp'):
            os.makedirs('output/tmp')
        idx = self.cfg.folder_name
        tmp_path = f'output/tmp/regressor_tmp_{idx}.pdb'
        with open(tmp_path, 'w') as f:
            f.write(pdbfile[0])

        # torchdrug-style inference
        proteins = [bio_load_pdb(tmp_path)[0]]
        protein = data.Protein.pack(proteins).cuda(self.device)
        batch = self.transform({"graph": protein})
        with torch.no_grad():
            pred = self.task.predict(batch)

        return pred.cpu()

    def calc_total_loss(self, x, cfg):
        """
            Easy one-stop-shop that calls out to all the implemented loss calculators,
            aggregates logs, and weights total_loss.
        """

        logs = {}
        total_loss = 0
        e_cfg = cfg.accept_reject.energy_cfg
        if e_cfg.fold_w:
            fold_m_nlls, plddt = self.calc_fold_loss(x, cfg.antigen, cfg.objects, cfg.limit_range,
                e_cfg.selection, e_cfg.reduction, e_cfg.num_recycles)
            fold_m_nlls = (fold_m_nlls * torch.tensor(e_cfg.fold_w)).sum()
            logs['fold_loss'] = fold_m_nlls.item()
            fold_conf = calc_fold_conf(x, plddt, cfg)
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
        if e_cfg.get('falsePOS_w', None):
            reg_loss = self.calc_regressor()
            logs['activity'] = reg_loss.item()
            falsePOS_loss = (reg_loss * torch.tensor(e_cfg.falsePOS_w)).sum()
            logs['falsePOS_loss'] = falsePOS_loss.item()
            total_loss += falsePOS_loss
        logs['total_loss'] = total_loss.item()

        return total_loss, logs  # [B], Dict[str:[B]]

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

        output = self.struct_model.infer([antigen + antibody], num_recycles=num_recycles)
        xyz, plddt = get_xyz(output)

        losses = []
        if mode == 'brute':
            for l in len_insterest:
                for i in range(len(antigen) - l):
                    losses.append([calc_distance(xyz, antigen, limit_range, [[i, i+l]]), i, i+l])
        elif mode == 'convexhull':
            hull = ConvexHull(xyz[0].cpu().numpy())
            for l in len_insterest:
                for i in range(len(antigen) - l):
                    if sum([i in hull.vertices for i in range(i,i+l)]) == 0:
                        continue
                    losses.append([calc_distance(xyz, antigen, limit_range, [[i, i+l]]), i, i+l])
        
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
        if self.cfg.task == 'fixedseqs_fold':
            stage_fixedseqs_fold(self, design_cfg)
        elif self.cfg.task == 'stage_Astar_fold':
            stage_Astar_fold(self, design_cfg)
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