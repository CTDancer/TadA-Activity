import pandas as pd
import numpy as np
import esm


def load_table(path):
    tb = pd.read_csv(path, sep='\t')
    bodies = []
    for i in range(len(tb)):
        ab = list(tb.iloc[i, 0])
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
    bs = 4
    start, end = 0, 10
    data = data[start: end]
    for i in range(0, len(data), bs):
        datax = [(j, data[j]) data[i : i + bs]]
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
            import pdb; pdb.set_trace()
    tag = f"{start}-{end}"
    np.save(f'output_17k/dict_repr.npy', res)
    print(f"Schedule:{start}-{end}")
    return res


def cluster(path):
    antibody = load_table(path)
    feature = extract_feature(antibody)
    import pdb; pdb.set_trace()




if __name__ == '__main__':
    path = 'data/patent_sequence.tsv'
    cluster(path)
