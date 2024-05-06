import pandas as pd
import numpy as np
import esm
import torch
from tqdm import tqdm
from sklearn.cluster import KMeans


def load_table(path, mask=False):
    tb = pd.read_csv(path, sep='\t')
    bodies = []
    for i in range(len(tb)):
        ab = list(tb.iloc[i, 0])
        if mask:
            for j in range(1, 4):
                s = tb.iloc[i, 0].index(tb.iloc[i, j])
                l = len(tb.iloc[i, j])
                ab[s : s + l] = ['<mask>'] * l
        bodies += [''.join(ab)]
    return bodies


def extract_feature(data):
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model = model.cuda()
    model.eval()

    res = {}
    start, end = 0, 18000
    end = min(end, len(data))
    print(f'[LOG] Aiming at ({start}-{end})/{len(data)}.')
    npz_path = f'output/cluster/dict_repr_{start}_{end}_mask.npy'
    print(npz_path)
    for i in tqdm(range(start, end)):
        datax = [(i, data[i])]
        batch_labels, batch_strs, batch_tokens = batch_converter(datax)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        
        with torch.no_grad():
            batch_tokens = batch_tokens.cuda()
            results = model(batch_tokens, repr_layers=[33])
        token_representations = results["representations"][33].detach().clone().cpu().numpy()  # torch.Size([4, 73, 1280])
        
        # Generate per-sequence representations via averaging
        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
        for i, tokens_len in enumerate(batch_lens):
            res[batch_labels[i]] = token_representations[i, 1 : tokens_len - 1].mean(0)
    tag = f"{start}-{end}"
    np.save(npz_path, res)
    print(f"Schedule:{start}-{end}")
    return res


def cluster(path):
    # antibody = load_table(path, mask=False)
    dist = np.load('output/cluster/dict_repr_0_17129.npy', allow_pickle=True).item()
    dist_item = sorted(dist.items(), key=lambda x: x[0])
    X = np.array([i[1] for i in dist_item])
    kmeans = KMeans(n_clusters=10, random_state=0).fit(X)
    # df = pd.DataFrame(dict(kmeans_10_mask=kmeans.labels_))
    # df.to_csv('output_17k/kmeans_10_mask.tsv', sep="\t", index=False)

    # import pdb; pdb.set_trace()
    from collections import Counter
    label = Counter(kmeans.labels_)
    print(label)
    # for i in 
    # antibody[]




if __name__ == '__main__':
    path = 'data/patent_sequence.tsv'
    # antibody = load_table(path, mask=True)
    # feature = extract_feature(antibody)
    cluster(path)
