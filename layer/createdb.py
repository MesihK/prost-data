from contactGroups import commp as cp
from esm1b import embed
from pickle import dump
import numpy as np
from time import time

allLayers = [ i for i in range(34)]
layers = {}
for i in range(34):
    layers[i] = []

seq = {}
for fasta in cp.fasta_iter('all.fa'):
    if fasta[0] in seq: continue
    seq[fasta[0]] = fasta[1]
    start = time()
    la = embed(fasta[1],allLayers,True)
    for i in range(34):
        layers[i].append(la[i])
    end = time()
    print(fasta[0],len(fasta[1]),(end-start),(end-start)/len(fasta[1])*1560181/60/60)


print('dumping seq')
with open('out/seq.pkl','wb') as f:
    dump(seq,f)

for i in range(34):
    print('dumping layer %d'%i)
    with open('out/layer.%d.pkl'%i,'wb') as f:
        print(np.shape(layers[i]))
        dump(layers[i],f)
