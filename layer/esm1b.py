import torch
import esm
import numpy as np

torch.set_num_threads(32)

# Load ESM-1b model
model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
batch_converter = alphabet.get_batch_converter()

def _embed(seq,layers,retAll):
    _, _, toks = batch_converter([("prot",seq)])
    with torch.no_grad():
        if torch.cuda.is_available():
             toks = toks.to(device="cuda", non_blocking=True)
        results = model(toks, repr_layers=layers)
        if not retAll:
            res = np.zeros((len(seq)+2,1280))
            for l in layers:
                res += results["representations"][l].to(device="cpu")[0].detach().numpy()
            return res/len(layers) 
        else:
            res = {}
            for l in layers:
                res[l] = results["representations"][l].to(device="cpu")[0].detach().numpy()
            return res

def embed(seq,layers=[33],retAll=False):
    l = len(seq)
    embtoks = None
    if l > 1022:
        piece = int(l/1022)+1
        part = l/piece
        for i in range(piece):
            st = int(i*part)
            sp = int((i+1)*part)
            results = _embed(seq[st:sp],layers,retAll)
            if not retAll:
                if embtoks is not None:
                    embtoks = np.concatenate((embtoks[:len(embtoks)-1],results[1:]),axis=0)
                else:
                    embtoks = results
            else:
                if embtoks is not None:
                    for l in layers:
                        embtoks[l] = np.concatenate((embtoks[l][:len(embtoks[l])-1],results[l][1:]),axis=0)
                else:
                    embtoks = results
                    
    else:
        embtoks = _embed(seq,layers,retAll)
    return embtoks
