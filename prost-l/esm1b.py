import torch
import esm
import numpy as np

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
    results = esm1b(toks, repr_layers=[13,25])
    
    l13 = results["representations"][13].to(device="cpu")[0].detach().numpy()
    l25 = results["representations"][25].to(device="cpu")[0].detach().numpy()
    
    lsoftmax = torch.nn.LogSoftmax(dim=1)
    logits = lsoftmax(results["logits"]).to(device="cpu")[0].detach().numpy()
    
    return l13,l25,logits


def embed(seq):
    l = len(seq)
    l13emb = None
    l25emb = None
    logits = None
    if l > 1022:
        piece = int(l/1022)+1
        part = l/piece
        for i in range(piece):
            st = int(i*part)
            sp = int((i+1)*part)
            l13,l25,rlogits = _embed(seq[st:sp])
            if l13emb is not None:
                l13emb = np.concatenate((l13emb[:len(l13emb)-1],l13[1:]),axis=0)
                l25emb = np.concatenate((l25emb[:len(l25emb)-1],l25[1:]),axis=0)
                logits = np.concatenate((logits[:len(logits)-1],rlogits[1:]),axis=0)
            else:
                l13emb = l13
                l25emb = l25
                logits = rlogits
    else:
        l13emb,l25emb,logits = _embed(seq)
    return l13emb,l25emb,logits

cache = {}
def perplexity(seq,logits):
    s = 0
    for i,a in enumerate(seq):
        p = logits[i+1][alphabet.get_idx(a)]
        s += p
    return s/len(seq)

from scipy.fftpack import dct, idct
import numpy as np

def iDCTquant(v,n):
    f = dct(v.T, type=2, norm='ortho')
    trans = idct(f[:,:n], type=2, norm='ortho')
    for i in range(len(trans)):
        trans[i] = scale(trans[i])
    return trans.T

def scale(v):
    M = np.max(v)
    m = np.min(v)
    return (v - m) / float(M - m)

def quant2D(emb,n=5,m=44):
    dct = iDCTquant(emb[1:len(emb)-1],n)
    dct = iDCTquant(dct.T,m).T
    dct = dct.reshape(n*m)
    return (dct*127).astype('int8')

def quantSeq(seq):
    l13,l25,logs = embed(seq.upper())
    #print(np.shape(l13),np.shape(l25),np.shape(logs))
    q25 = quant2D(l25,3,1280)
    q13 = quant2D(l13,3,1280)
    #print(np.shape(q13),np.shape(q25))
    return np.concatenate([q25,q13]),logs