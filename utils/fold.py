import numpy as np
import torch

from openfold.utils.feats import atom14_to_atom37


def calc_fold_conf(x, plddt, cfg):
    conf = []
    l_ag = [len(i) for i in cfg.antigen]
    for a in range(len(l_ag)):
        assert plddt.shape[1] == l_ag[a] + len(x)
        if cfg.get('focus_on_antigen', False):
            idx = list(range(*cfg.objects[a]))
        elif cfg.get('focus', None):
            idx = list(range(*cfg.focus[a]))
        else:
            idx = list(range(l_ag[a] + cfg.len_linker, l_ag[a] + len(x))) + \
                list(range(*cfg.objects[a]))
        confidence = plddt[a, idx].mean() / 100
        conf.append(confidence)
    return torch.tensor(conf)


def calc_distance(xyz, antigen, limit_range, objects, selection='min', reduction='mean'):
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


def get_xyz(output):
    # average on all atoms of a protein
    idx = output["atom37_atom_exists"]
    idx_sum = idx.sum(dim=2)
    plddt = (output["plddt"] * idx).sum(dim=2) / idx_sum
    xyz = atom14_to_atom37(output["positions"][-1], output)  # [B, L, 3]
    xyz = (xyz * idx[..., None].repeat(1,1,1,3)).sum(dim=2) / idx_sum[..., None].repeat(1,1,3)
    return xyz, plddt