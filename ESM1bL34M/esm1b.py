import torch
import esm
import numpy as np
torch.set_num_threads(32)

# Load ESM-1b model
esm1b, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
batch_converter = alphabet.get_batch_converter()
if torch.cuda.is_available():
    esm1b = esm1b.cuda()
    
for param in esm1b.parameters():
    param.grad = None
    param.requires_grad = False

def _embed(seq):
    _, _, toks = batch_converter([("prot",seq)])
    if torch.cuda.is_available():
        toks = toks.to(device="cuda", non_blocking=True)
    results = esm1b(toks, repr_layers=[33])
    
    l33 = results["representations"][33].to(device="cpu")[0].detach().numpy()
    
    lsoftmax = torch.nn.LogSoftmax(dim=1)
    logits = lsoftmax(results["logits"]).to(device="cpu")[0].detach().numpy()
    
    return l33,logits


def embed(seq):
    l = len(seq)
    embtoks = None
    logits = None
    if l > 1022:
        piece = int(l/1022)+1
        part = l/piece
        for i in range(piece):
            st = int(i*part)
            sp = int((i+1)*part)
            results,rlogits = _embed(seq[st:sp])
            if embtoks is not None:
                embtoks = np.concatenate((embtoks[:len(embtoks)-1],results[1:]),axis=0)
                logits = np.concatenate((logits[:len(logits)-1],rlogits[1:]),axis=0)
            else:
                embtoks = results
                logits = rlogits
    else:
        embtoks,logits = _embed(seq)
    return embtoks,logits

cache = {}
def perplexity(seq):
    if seq not in cache:
        emb,logits = embed(seq)
        cache[seq] = (emb,logits)
    else:
        emb,logits = cache[seq]
    s = 0
    for i,a in enumerate(seq):
        p = logits[i+1][alphabet.get_idx(a)]
        s += p
    return s/len(seq)
