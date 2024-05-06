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
import pdb
from torchdrug import core, models, tasks, datasets, utils, data, layers
from torchdrug.layers import functional
from torchdrug.core import Registry as R
from .network.parsers import parse_a3m, read_templates
from .network.RoseTTAFoldModel  import RoseTTAFoldModule_e2e, RoseTTAFoldModule_e2e_msaonly
from .network.trFold  import TRFold
from .network.ffindex import *
from .network.kinematics import xyz_to_c6d, c6d_to_bins2, xyz_to_t2d
from .network  import util

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
        # pdb.set_trace()
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
        # pdb.set_trace()
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

NBIN = [37, 37, 37, 19]

MODEL_PARAM ={
        "n_module"     : 8,
        "n_module_str" : 4,
        "n_module_ref" : 4,
        "n_layer"      : 1,
        "d_msa"        : 384 ,
        "d_pair"       : 288,
        "d_templ"      : 64,
        "n_head_msa"   : 12,
        "n_head_pair"  : 8,
        "n_head_templ" : 4,
        "d_hidden"     : 64,
        "r_ff"         : 4,
        "n_resblock"   : 1,
        "p_drop"       : 0.0,
        "use_templ"    : True,
        "performer_N_opts": {"nb_features": 64},
        "performer_L_opts": {"nb_features": 64}
        }

SE3_param = {
        "num_layers"    : 2,
        "num_channels"  : 16,
        "num_degrees"   : 2,
        "l0_in_features": 32,
        "l0_out_features": 8,
        "l1_in_features": 3,
        "l1_out_features": 3,
        "num_edge_features": 32,
        "div": 2,
        "n_heads": 4
        }

REF_param = {
        "num_layers"    : 3,
        "num_channels"  : 32,
        "num_degrees"   : 3,
        "l0_in_features": 32,
        "l0_out_features": 8,
        "l1_in_features": 3,
        "l1_out_features": 3,
        "num_edge_features": 32,
        "div": 4,
        "n_heads": 4
        }
MODEL_PARAM['SE3_param'] = SE3_param
MODEL_PARAM['REF_param'] = REF_param

# params for the folding protocol
fold_params = {
    "SG7"     : np.array([[[-2,3,6,7,6,3,-2]]])/21,
    "SG9"     : np.array([[[-21,14,39,54,59,54,39,14,-21]]])/231,
    "DCUT"    : 19.5,
    "ALPHA"   : 1.57,
    
    # TODO: add Cb to the motif
    "NCAC"    : np.array([[-0.676, -1.294,  0.   ],
                          [ 0.   ,  0.   ,  0.   ],
                          [ 1.5  , -0.174,  0.   ]], dtype=np.float32),
    "CLASH"   : 2.0,
    "PCUT"    : 0.5,
    "DSTEP"   : 0.5,
    "ASTEP"   : np.deg2rad(10.0),
    "XYZRAD"  : 7.5,
    "WANG"    : 0.1,
    "WCST"    : 0.1
}

fold_params["SG"] = fold_params["SG9"]
        
@R.register("model.RoseTTAFold")
class RoseTTAFold(nn.Module, core.Configurable):
    def __init__(self, path, model="e2e", readout="mean"):
        # if torch.cuda.is_available():
        #     self.device = torch.device("cuda:0")
        # else:
        #     self.device = torch.device("cpu")
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        super(RoseTTAFold, self).__init__()
        self.model = RoseTTAFoldModule_e2e_msaonly(**MODEL_PARAM).to(device)
        self.active_fn = nn.Softmax(dim=1)
        self.output_dim = MODEL_PARAM["d_msa"]
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        # self.data_path = data_path
        # self.label_file = self.data_path + label_file
        
        # self.seq2name = self.get_name_seq_dict(self.label_file)
        # pdb.set_trace()
        
        could_load = self.load_model(path, model, device)
        if not could_load:
            print ("ERROR: failed to load model")
            sys.exit()
            
        if readout == "sum":
            self.readout = layers.SumReadout("residue")
        elif readout == "mean":
            self.readout = layers.MeanReadout("residue")
        else:
            raise ValueError("Unknown readout `%s`" % readout)
            
    def get_name_seq_dict(self, label_file):
        with open(label_file, 'r') as file:
            lines = [line.strip() for line in file.readlines()][3:]
        seq2name = {}
        for line in lines:
            name, seq, _ = line.strip().split(',')
            seq2name[seq] = name
        return seq2name
        
    def load_model(self, path, model, device):
        chk_fn = path + f"RoseTTAFold_{model}.pt"
        if not os.path.exists(chk_fn):
            return False
        checkpoint = torch.load(chk_fn, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        return True
        
    def extend(self, a,b,c, L,A,D):
        '''
        input:  3 coords (a,b,c), (L)ength, (A)ngle, and (D)ihedral
        output: 4th coord
        '''
        N = lambda x: x/np.sqrt(np.square(x).sum(-1,keepdims=True) + 1e-8)
        bc = N(b-c)
        n = N(np.cross(b-a, bc))
        m = [bc,np.cross(n,bc),n]
        d = [L*np.cos(A), L*np.sin(A)*np.cos(D), -L*np.sin(A)*np.sin(D)]
        return c + sum([m*d for m,d in zip(m,d)])

    def write_pdb(self, seq, atoms, idx, Bfacts=None, prefix=None):
        L = len(seq)
        filename = "%s.pdb"%prefix
        ctr = 1
        with open(filename, 'wt') as f:
            if Bfacts == None:
                Bfacts = np.zeros(L)
            else:
                Bfacts = torch.clamp( Bfacts, 0, 1)
            
            for i,s in enumerate(seq):
                if (len(atoms.shape)==2):
                    f.write ("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"%(
                            "ATOM", ctr, " CA ", util.num2aa[s], 
                            "A", idx[i]+1, atoms[i,0], atoms[i,1], atoms[i,2],
                            1.0, Bfacts[i] ) )
                    ctr += 1

                elif atoms.shape[1]==3:
                    for j,atm_j in enumerate((" N  "," CA "," C  ")):
                        f.write ("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"%(
                                "ATOM", ctr, atm_j, util.num2aa[s], 
                                "A", idx[i]+1, atoms[i,j,0], atoms[i,j,1], atoms[i,j,2],
                                1.0, Bfacts[i] ) )
                        ctr += 1                
                
                elif atoms.shape[1]==4:
                    for j,atm_j in enumerate((" N  "," CA "," C  ", " O  ")):
                        f.write ("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"%(
                                "ATOM", ctr, atm_j, util.num2aa[s], 
                                "A", idx[i]+1, atoms[i,j,0], atoms[i,j,1], atoms[i,j,2],
                                1.0, Bfacts[i] ) )
                        ctr += 1                
        
    def forward(self, graph, inputs, all_loss=None, metric=None):
        # print("forward")
        window=150
        shift=75
        # seq = protein.residue_type
        # seq = ''.join([torchdrug.data.Protein.id2residue_symbol[seq[i].item()] for i in range(len(seq))])
        # # name = self.seq2name[seq]
        # workdir = self.data_path + f"/RF_results/"
        # if not os.path.exists(workdir):
        #     os.makedirs(workdir)
        # fasta_str = f">{len(seq)} residues|\n{seq}"
        # fasta_pth = workdir + f"seq.fasta"
        # with open(fasta_pth, 'w') as file:
        #     file.write(fasta_str)
            
        #     # hhblits
        # script_path = "/home/ubuntu/scratch/tongchen/Protein-Activity-Prediction/RoseTTAFold/run_e2e_ver_2.sh"
        # result = subprocess.run([script_path, fasta_pth, workdir], capture_output=True, text=True)

        # a3m_fn = workdir + "t000_.msa0.a3m"
        # hhr_fn = workdir + "t000_.hhr"
        # atab_fn = workdir + "t000_.atab" 
        # print("done hhblits")
        # pdb.set_trace()
        # start = time.time()
        size = graph.batch_size
        msas = []
        for i in range(size):
            a3m_fn = inputs["a3m_fn"][i]
            # hhr_fn = inputs["hhr_fn"][i]
            hhr_fn = None
            atab_fn = inputs["atab_fn"][i]
            
            device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
            FFDB = '/home/ubuntu/scratch/tongchen/Protein-Activity-Prediction/RoseTTAFold/pdb100_2021Mar03/pdb100_2021Mar03'
            FFindexDB = namedtuple("FFindexDB", "index, data")
            ffdb = FFindexDB(read_index(FFDB+'_pdb.ffindex'), read_data(FFDB+'_pdb.ffdata'))
            
            msa = parse_a3m(a3m_fn)
            N, L = msa.shape
            #
            if hhr_fn != None:
                xyz_t, t1d, t0d = read_templates(L, ffdb, hhr_fn, atab_fn, n_templ=10)
            else:
                xyz_t = torch.full((1, L, 3, 3), np.nan).float()
                t1d = torch.zeros((1, L, 3)).float()
                t0d = torch.zeros((1,3)).float()
            #
            msa = torch.tensor(msa).long().view(1, -1, L)
            idx_pdb = torch.arange(L).long().view(1, L)
            seq = msa[:,0]
            #
            # template features
            xyz_t = xyz_t.float().unsqueeze(0)
            t1d = t1d.float().unsqueeze(0)
            t0d = t0d.float().unsqueeze(0)
            t2d = xyz_to_t2d(xyz_t, t0d)
            
            self.model.eval()
            with torch.no_grad():
                # do cropped prediction if protein is too big
                if L > window*2:
                    prob_s = [np.zeros((L,L,NBIN[i]), dtype=np.float32) for  i in range(4)]
                    count_1d = np.zeros((L,), dtype=np.float32)
                    count_2d = np.zeros((L,L), dtype=np.float32)
                    node_s = np.zeros((L,MODEL_PARAM['d_msa']), dtype=np.float32)
                    #
                    grids = np.arange(0, L-window+shift, shift)
                    ngrids = grids.shape[0]
                    print("ngrid:     ", ngrids)
                    print("grids:     ", grids)
                    print("windows:   ", window)

                    for i in range(ngrids):
                        for j in range(i, ngrids):
                            start_1 = grids[i]
                            end_1 = min(grids[i]+window, L)
                            start_2 = grids[j]
                            end_2 = min(grids[j]+window, L)
                            sel = np.zeros((L)).astype(np.bool)
                            sel[start_1:end_1] = True
                            sel[start_2:end_2] = True
                        
                            input_msa = msa[:,:,sel]
                            mask = torch.sum(input_msa==20, dim=-1) < 0.5*sel.sum() # remove too gappy sequences
                            input_msa = input_msa[mask].unsqueeze(0)
                            input_msa = input_msa[:,:1000].to(device)
                            input_idx = idx_pdb[:,sel].to(device)
                            input_seq = input_msa[:,0].to(device)
                            #
                            # Select template
                            input_t1d = t1d[:,:,sel].to(device) # (B, T, L, 3)
                            input_t2d = t2d[:,:,sel][:,:,:,sel].to(device)
                            #
                            print ("running crop: %d-%d/%d-%d"%(start_1, end_1, start_2, end_2), input_msa.shape)
                            with torch.cuda.amp.autocast():
                                logit_s, node, init_crds, pred_lddt = self.model(input_msa, input_seq, input_idx, t1d=input_t1d, t2d=input_t2d, return_raw=True)
                            #
                            # Not sure How can we merge init_crds.....
                            sub_idx = input_idx[0].cpu()
                            sub_idx_2d = np.ix_(sub_idx, sub_idx)
                            count_2d[sub_idx_2d] += 1.0
                            count_1d[sub_idx] += 1.0
                            node_s[sub_idx] += node[0].cpu().numpy()
                            for i_logit, logit in enumerate(logit_s):
                                prob = self.active_fn(logit.float()) # calculate distogram
                                prob = prob.squeeze(0).permute(1,2,0).cpu().numpy()
                                prob_s[i_logit][sub_idx_2d] += prob
                            del logit_s, node
                    #
                    # combine all crops
                    for i in range(4):
                        prob_s[i] = prob_s[i] / count_2d[:,:,None]
                    prob_in = np.concatenate(prob_s, axis=-1)
                    node_s = node_s / count_1d[:, None]
                    #
                    # Do iterative refinement using SE(3)-Transformers
                    # clear cache memory
                    torch.cuda.empty_cache()
                    #
                    node_s = torch.tensor(node_s).to(device).unsqueeze(0)
                    # seq = msa[:,0].to(device)
                    # idx_pdb = idx_pdb.to(device)
                    # prob_in = torch.tensor(prob_in).to(device).unsqueeze(0)
                    # with torch.cuda.amp.autocast():
                    #     xyz, lddt = self.model(node_s, seq, idx_pdb, prob_s=prob_in, refine_only=True)
                else:
                    msa = msa[:,:1000].to(device)
                    seq = msa[:,0]
                    idx_pdb = idx_pdb.to(device)
                    t1d = t1d[:,:10].to(device)
                    t2d = t2d[:,:10].to(device)
                    with torch.cuda.amp.autocast():
                        node_s = self.model(msa, seq, idx_pdb, t1d=t1d, t2d=t2d) # changed _ to node_s
                    # prob_s = list()
                    # for logit in logit_s:
                    #     prob = self.active_fn(logit.float()) # distogram
                    #     prob = prob.reshape(-1, L, L).permute(1,2,0).cpu().numpy()
                    #     prob_s.append(prob)
            msas.append(node_s[0])    
        
        # pdb.set_trace()
        residue_feature = torch.cat(msas, dim=0)
        graph_feature = self.readout(graph, residue_feature)
        
        # print(f"time = {time.time() - start}s")
        return {
            "graph_feature": graph_feature,
            "residue_feature": residue_feature
        }