import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from sklearn.metrics import precision_recall_curve, auc

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

def load(fpath):
    with open(fpath,'r') as f:
        score = []
        real = []
        for l in f:
            l = l.rstrip().split()
            score.append(-float(l[2]))
            real.append(float(l[3]))
        return score,real

def stats(res):
    score,real = res
    aucScore = roc_auc_score(real, score)
    auprc = average_precision_score(real, score)
    auc1k = aucNth(real,score,1000)
    thr, f1sc = f1(real,score)
    return '%.1f %.1f %.1f %.1f %.1f'%(aucScore*100,auprc*100,auc1k*100,f1sc*100,thr)

def run(layer):
    pfamr = stats(load('results/pfam.%d.5.44.res'%layer))
    gener = stats(load('results/gene.%d.5.44.res'%layer))
    supfr = stats(load('results/supf.%d.5.44.res'%layer))
    return pfamr+' '+gener+' '+supfr

for i in range(34):
    print(i,run(i))
