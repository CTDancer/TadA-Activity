import os
import sys
import glob
import warnings
from tqdm import tqdm
import jinja2
import yaml
import easydict
from collections import defaultdict
import torch
from torch import nn
import numpy as np

from torchdrug import core, models, tasks, datasets, utils, data
from torchdrug.layers import functional
from torchdrug.core import Registry as R

from Bio.PDB import PDBParser

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def bio_load_pdb(pdb_file):
    parser = PDBParser(QUIET=True)
    protein = parser.get_structure(0, pdb_file)
    residues = [residue for residue in protein.get_residues()]
    residue_type = [data.Protein.residue2id[residue.get_resname()] for residue in residues]
    residue_number = [residue.full_id[3][1] for residue in residues]
    id2residue = {residue.full_id: i for i, residue in enumerate(residues)}
    residue_feature = functional.one_hot(torch.as_tensor(residue_type), len(data.Protein.residue2id)+1)

    atoms = [atom for atom in protein.get_atoms()]
    atoms = [atom for atom in atoms if atom.get_name() in data.Protein.atom_name2id]
    occupancy = [atom.get_occupancy() for atom in atoms]
    b_factor = [atom.get_bfactor() for atom in atoms]
    atom_type = [data.feature.atom_vocab.get(atom.get_name()[0], 0) for atom in atoms]
    atom_name = [data.Protein.atom_name2id.get(atom.get_name(), 37) for atom in atoms]
    node_position = np.stack([atom.get_coord() for atom in atoms], axis=0)
    node_position = torch.as_tensor(node_position)
    atom2residue = [id2residue[atom.get_parent().full_id] for atom in atoms]

    edge_list = [[0, 0, 0]]
    bond_type = [0]

    return data.Protein(edge_list, atom_type=atom_type, bond_type=bond_type, residue_type=residue_type,
                num_node=len(atoms), num_residue=len(residues), atom_name=atom_name, 
                atom2residue=atom2residue, occupancy=occupancy, b_factor=b_factor,
                residue_number=residue_number, node_position=node_position, residue_feature=residue_feature
            ), "".join([data.Protein.id2residue_symbol[res] for res in residue_type])


@R.register("datasets.Vactivity")
class Vactivity(data.ProteinDataset):

    processed_file = "Vactivity.pkl.gz"

    def __init__(self, path, split_ratio=(0.6, 0.2, 0.2), discrete_label=False, verbose=1, label_file='20230927tada_balanced.csv', **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        self.split_ratio = split_ratio
        self.discrete_label = discrete_label
        
        pkl_file = os.path.join(path, self.processed_file)

        if os.path.exists(pkl_file):
            self.load_pickle(pkl_file, verbose=verbose, **kwargs)
        else:
            pdb_files = sorted(glob.glob(os.path.join(path, 'pdb', "*.pdb")))
            self.load_pdbs(pdb_files, verbose=verbose, **kwargs)
            self.save_pickle(pkl_file, verbose=verbose)

        # for balance
        self.name2data = {os.path.basename(self.pdb_files[i])[:-4]: i for i in range(len(self.pdb_files))}
        label_file = os.path.join(path, label_file)
        self.label_list, self.label_dict = self.get_label_list(label_file)
        activitity = [self.label_dict[name] for name in self.label_list]
        self.targets = {"activity": activitity}
        if self.discrete_label:
            num_labels = defaultdict(int)
            for i in activitity:
                num_labels[i] += 1
            print(num_labels)

    def load_pdbs(self, pdb_files, transform=None, lazy=False, verbose=0, **kwargs):
        num_sample = len(pdb_files)
        if num_sample > 1000000:
            warnings.warn("Preprocessing proteins of a large dataset consumes a lot of CPU memory and time. "
                          "Use load_pdbs(lazy=True) to construct molecules in the dataloader instead.")

        self.transform = transform
        self.lazy = lazy
        self.kwargs = kwargs
        self.data = []
        self.pdb_files = []
        self.sequences = []

        if verbose:
            pdb_files = tqdm(pdb_files, "Constructing proteins from pdbs")
        for i, pdb_file in enumerate(pdb_files):
            protein, sequence = bio_load_pdb(pdb_file)
            if hasattr(protein, "residue_feature"):
                with protein.residue():
                    protein.residue_feature = protein.residue_feature.to_sparse()
            self.data.append(protein)
            self.pdb_files.append(pdb_file)
            self.sequences.append(sequence)

    def get_label_list(self, label_file):
        with open(label_file, "r") as fin:
            lines = [line.strip() for line in fin.readlines()][2:]
        label_dict = {}
        label_list = []
        for line in lines:
            name, sequence, activity = line.split(",")
            activity = float(activity)
            if self.discrete_label:
                label_dict[name] = (activity > 5)
            else:
                label_dict[name] = activity
            label_list.append(name)
        return label_list, label_dict

    def split(self, split_ratio=None):
        split_ratio = split_ratio or self.split_ratio
        num_samples = [int(len(self) * ratio) for ratio in split_ratio]
        num_samples[-1] = len(self) - sum(num_samples[:-1])
        splits = torch.utils.data.random_split(self, num_samples)
        return splits
    
    def get_item(self, index):
        name = self.label_list[index]
        index = self.name2data[name]
        if self.lazy:
            protein = data.Protein.from_pdb(self.pdb_files[index], self.kwargs)
        else:
            protein = self.data[index].clone()
        if hasattr(protein, "residue_feature"):
            with protein.residue():
                protein.residue_feature = protein.residue_feature.to_dense()
        item = {"graph": protein, "activity": self.label_dict[name]}
        if self.transform:
            item = self.transform(item)
        return item
    
    def __len__(self):
        return len(self.label_list)


@R.register("datasets.V5")
class V5(data.ProteinDataset):

    processed_file = "v5.pkl.gz"

    def __init__(self, path, split_ratio=(0.6, 0.2, 0.2), verbose=1, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        self.split_ratio = split_ratio

        pkl_file = os.path.join(path, self.processed_file)

        if os.path.exists(pkl_file):
            self.load_pickle(pkl_file, verbose=verbose, **kwargs)
        else:
            pdb_files = sorted(glob.glob(os.path.join(path, "*.pdb")))
            self.load_pdbs(pdb_files, verbose=verbose, **kwargs)
            self.save_pickle(pkl_file, verbose=verbose)

        # label_file = os.path.join(path, "data_5V.csv")
        # label_list = self.get_label_list(label_file)
        # activitity = [label_list[os.path.basename(pdb_file)[:-4]] for pdb_file in self.pdb_files]
        # self.targets = {"activity": activitity}

    def load_pdbs(self, pdb_files, transform=None, lazy=False, verbose=0, **kwargs):
        num_sample = len(pdb_files)
        if num_sample > 1000000:
            warnings.warn("Preprocessing proteins of a large dataset consumes a lot of CPU memory and time. "
                          "Use load_pdbs(lazy=True) to construct molecules in the dataloader instead.")

        self.transform = transform
        self.lazy = lazy
        self.kwargs = kwargs
        self.data = []
        self.pdb_files = []
        self.sequences = []

        if verbose:
            pdb_files = tqdm(pdb_files, "Constructing proteins from pdbs")
        for i, pdb_file in enumerate(pdb_files):
            protein, sequence = bio_load_pdb(pdb_file)
            if hasattr(protein, "residue_feature"):
                with protein.residue():
                    protein.residue_feature = protein.residue_feature.to_sparse()
            self.data.append(protein)
            self.pdb_files.append(pdb_file)
            self.sequences.append(sequence)

    def get_label_list(self, label_file):
        with open(label_file, "r") as fin:
            lines = [line.strip() for line in fin.readlines()][2:]
        label_list = {}
        for line in lines:
            name, sequence, activity = line.split(",")
            activity = float(activity)
            label_list[name] = activity
        return label_list

    def split(self, split_ratio=None):
        split_ratio = split_ratio or self.split_ratio
        num_samples = [int(len(self) * ratio) for ratio in split_ratio]
        num_samples[-1] = len(self) - sum(num_samples[:-1])
        splits = torch.utils.data.random_split(self, num_samples)
        return splits
    
    def get_item(self, index):
        if self.lazy:
            protein = data.Protein.from_pdb(self.pdb_files[index], self.kwargs)
        else:
            protein = self.data[index].clone()
        if hasattr(protein, "residue_feature"):
            with protein.residue():
                protein.residue_feature = protein.residue_feature.to_dense()
        item = {"graph": protein, "activity": self.targets["activity"][index]}
        if self.transform:
            item = self.transform(item)
        return item


@R.register("datasets.falsePOS")
class falsePOS(V5):
    
    processed_file = "falsePOS.pkl.gz"

    def __init__(self, path, split_ratio=(0.8, 0.1, 0.1), verbose=1, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        self.split_ratio = split_ratio

        pkl_file = os.path.join(path, self.processed_file)
        
        pdb_files = glob.glob(os.path.join(path, 'pdb', "*.pdb"))
        pdb_files = sorted(pdb_files)
        score_dict = [-1] * len(pdb_files)

        # select a fraction
        num_pos = len(pdb_files) // 3
        pdb_files += num_pos * [os.path.join(path, "ESM5.pdb")]
        score_dict += [1] * num_pos
        
        if os.path.exists(pkl_file):
            self.load_pickle(pkl_file, verbose=verbose, **kwargs)
        else:
            self.load_pdbs(pdb_files, verbose=verbose, **kwargs)
            self.save_pickle(pkl_file, verbose=verbose)
        self.targets = {"activity": score_dict}

    def get_score_dict(self, score_file):
        with open(score_file, "r") as fin:
            lines = [line.strip() for line in fin.readlines()][1:]
        score_dict = {}
        for line in lines:
            name, sequence, distance, conf, activity = line.split("\t")
            score_dict['17k_' + name] = float(activity)
        score_dict['ESM5'] = 80
        return score_dict


    
def load_config(cfg_file, context={}):
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    template = jinja2.Template(raw)
    instance = template.render(context)
    cfg = yaml.safe_load(instance)
    cfg = easydict.EasyDict(cfg)
    return cfg


@R.register("models.FusionNetwork")
class FusionNetwork(nn.Module, core.Configurable):

    def __init__(self, sequence_model, structure_model):
        super(FusionNetwork, self).__init__()
        self.sequence_model = sequence_model
        self.structure_model = structure_model
        self.output_dim = sequence_model.output_dim + structure_model.output_dim

    def forward(self, graph, input, all_loss=None, metric=None):
        output1 = self.sequence_model(graph, input, all_loss, metric)
        node_output1 = output1.get("node_feature", output1.get("residue_feature"))
        output2 = self.structure_model(graph, node_output1, all_loss, metric)
        node_output2 = output2.get("node_feature", output2.get("residue_feature"))
        
        node_feature = torch.cat([node_output1, node_output2], dim=-1)
        graph_feature = torch.cat([
            output1['graph_feature'], 
            output2['graph_feature']
        ], dim=-1)

        return {
            "graph_feature": graph_feature,
            "node_feature": node_feature
        }
    