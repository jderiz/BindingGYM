import torch
import numpy as np
import itertools

def shuffle_subset(n, p):
    n_shuffle = np.random.binomial(n, p)
    ix = np.arange(n)
    ix_subset = np.random.choice(ix, size=n_shuffle, replace=False)
    ix_subset_shuffled = np.copy(ix_subset)
    np.random.shuffle(ix_subset_shuffled)
    ix[ix_subset] = ix_subset_shuffled
    return ix

def featurize_GTrans(batch, shuffle_fraction=0.):
    """ Pack and pad batch into torch tensors """
    alphabet = 'ACDEFGHIKLMNPQRSTVWY-'
    B = len(batch)
    lengths = np.array([len(b['seq']) for b in batch], dtype=np.int32)
    L_max = max([len(b['seq']) for b in batch])
    X = np.zeros([B, L_max, 4, 3])
    S = np.zeros([B, L_max], dtype=np.int32)
    score = np.ones([B, L_max]) * 100.0

    # Build the batch
    for i, b in enumerate(batch):
        print(b['seq'])
        x = np.stack([b[c] for c in ['N', 'CA', 'C', 'O']], 1) # [#atom, 4, 3]
        
        l = len(b['seq'])
        x_pad = np.pad(x, [[0,L_max-l], [0,0], [0,0]], 'constant', constant_values=(np.nan, )) # [#atom, 4, 3]
        X[i,:,:,:] = x_pad

        # Convert to labels
        indices = np.asarray([alphabet.index(a) for a in b['seq']], dtype=np.int32)
        if shuffle_fraction > 0.:
            idx_shuffle = shuffle_subset(l, shuffle_fraction)
            S[i, :l] = indices[idx_shuffle]
        else:
            S[i, :l] = indices
    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32)
    seq_mask = (S<20).astype(np.float32)
    mask = mask * seq_mask

    # * (S>19).astype(float) # atom mask
    numbers = np.sum(mask, axis=1).astype(int)
    S_new = np.zeros_like(S)
    X_new = np.zeros_like(X)+np.nan
    for i, n in enumerate(numbers):
        X_new[i,:n,::] = X[i][mask[i]==1]
        S_new[i,:n] = S[i][mask[i]==1]

    X = X_new
    S = S_new
    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32)
    X[isnan] = 0.
    # Conversion
    S = torch.from_numpy(S).to(dtype=torch.long)
    score = torch.from_numpy(score).float()
    X = torch.from_numpy(X).to(dtype=torch.float32)
    mask = torch.from_numpy(mask).to(dtype=torch.float32)
    return X, S, score, mask, lengths

