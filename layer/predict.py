import re
from pickle import load
from scipy.fftpack import dct, idct
import numpy as np
from time import time
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from sklearn.metrics import precision_recall_curve, auc
from multiprocessing import Pool
import os

def f1(real, score):
    precision, recall, thresholds = precision_recall_curve(real, score)
    f1_scores = 2*recall*precision/(recall+precision)
    return thresholds[np.argmax(f1_scores)],np.max(f1_scores)

def aucNth(y, yp, N):
    assert len(y) == len(yp)
    assert len(y) > 1
    fpr, tpr, thresholds = roc_curve(y, yp)
    negatives = y.count(0)
    assert N < negatives
    perc = N / float(negatives)
    fpr1k = []
    tpr1k = []
    i = 0
    while i < len(fpr):
        if fpr[i] > perc:
            break
        fpr1k.append(fpr[i])
        tpr1k.append(tpr[i])
        i+=1
    assert len(fpr1k) > 1
    aucScore = auc(fpr1k, tpr1k) / perc
    return aucScore

def dctquant(v,n):
    f = dct(v.T, type=2, norm='ortho')
    trans = idct(f[:,:n], type=2, norm='ortho')
    for i in range(len(trans)):
        trans[i] = scale(trans[i])
    return trans.T

def scale(v):
    M = np.max(v)
    m = np.min(v)
    return (v - m) / float(M - m)

def quant(emb,n=5,m=44):
    dct = dctquant(emb[1:len(emb)-1],n)
    ddct = dctquant(dct.T,m).T
    return (ddct*127).astype('int8')

with open('out/seq.pkl','rb') as f:
    seq = load(f)
seqlist = list(seq.keys())

layer = 32
layers = {}
def load_layer(layer):
    global layers
    with open('out/layer.%d.pkl'%layer,'rb') as f:
        layers[layer] = load(f)

def get_emb(prot,l):
    return layers[l][seqlist.index(prot)]

def predict(p1,p2,layer):
    q1 = quant(get_emb(p1,layer))
    q2 = quant(get_emb(p2,layer))
    return abs(q1-q2).sum(axis=1).sum()

phom = '../benchmark/Methods_benchmarking_pairs/pfam_hom-pairs_max50.txt'
pnon = '../benchmark/Methods_benchmarking_pairs/pfam_nonhom-pairs_max50.txt'
ghom = '../benchmark/Methods_benchmarking_pairs/gene3d_hom-pairs_max50.txt'
gnon = '../benchmark/Methods_benchmarking_pairs/gene3d_nonhom-pairs_max50.txt'
shom = '../benchmark/Methods_benchmarking_pairs/supfam_hom-pairs_max50.txt'
snon = '../benchmark/Methods_benchmarking_pairs/supfam_nonhom-pairs_max50.txt'

def read_benchmark_pairs(filepath,isHomolog):
    pairs = list()
    with open(filepath,'r') as f:
        for line in f:
            pair = []
            for searchResult in re.finditer('[@_]([A-Za-z]\d[A-Z\d]+)',line):
                res = searchResult.groups()[0]
                pair.append(res)
            pairs.append((pair[0],pair[1],isHomolog))
    return pairs

phoml = read_benchmark_pairs(phom,1)
pnonl = read_benchmark_pairs(pnon,0)
ghoml = read_benchmark_pairs(ghom,1)
gnonl = read_benchmark_pairs(gnon,0)
shoml = read_benchmark_pairs(shom,1)
snonl = read_benchmark_pairs(snon,0)

def pred(pair):
    global layer
    p1,p2,hom = pair
    dist = predict(p1,p2,layer)
    return [p1,p2,dist,hom]

def stats(res):
    real = []
    score = []
    for r in res:
        real.append(float(r[3]))
        score.append(-float(r[2]))
    aucScore = roc_auc_score(real, score)
    auprc = average_precision_score(real, score)
    auc1k = aucNth(real,score,1000)
    thr, f1sc = f1(real,score)
    print('%d %.1f %.1f %.1f %.1f %.1f'%(layer,aucScore*100,auprc*100,auc1k*100,f1sc*100,thr))

def write_res(res,fname):
    with open(fname,'w') as f:
        for r in res:
            f.write('%s %s %d %d \n'%(r[0],r[1],r[2],r[3]))

def benchmark():
    global layer, layers
    layers = {}
    load_layer(layer)
    pool = Pool(os.cpu_count())
    phomr = pool.map(pred, phoml)
    pnonr = pool.map(pred, pnonl)
    ghomr = pool.map(pred, ghoml)
    gnonr = pool.map(pred, gnonl)
    shomr = pool.map(pred, shoml)
    snonr = pool.map(pred, snonl)
    pfamr = phomr+pnonr
    gener = ghomr+gnonr
    supfr = shomr+snonr
    allr = pfamr+gener+supfr
    stats(allr)
    write_res(pfamr,'results/pfam.%d.5.44.res'%layer)
    write_res(gener,'results/gene.%d.5.44.res'%layer)
    write_res(supfr,'results/supf.%d.5.44.res'%layer)
    write_res(allr, 'results/allr.%d.5.44.res'%layer)

print("CPU",os.cpu_count())
for i in reversed(range(34)):
    layer = i
    benchmark()

