import re
from itertools import groupby
from pickle import load,loads,dump,dumps
import blosc

from esm1b import embed

def fasta_iter(fastafile):
    fh = open(fastafile)
    faiter = (x[1] for x in groupby(fh, lambda line: line[0] == ">"))
    for header in faiter:
        header = next(header)[1:].strip()
        seq = "".join(s.strip() for s in next(faiter))
        yield header, seq

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()
        
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


def read_fasta(fasta):
    seq = {}
    for f in fasta_iter(fasta):
        if f[0] not in seq:
            seq[f[0]] = f[1]
    return seq

def embed_seq(seq):
    emb = {}
    logs = {}
    total = len(seq)
    for i,s in enumerate(seq.keys()):
        e,l  = embed(seq[s])
        emb[s] = e
        logs[s] = l
        printProgressBar(i,total,'Embedding','Complate',length=50)
    return emb,logs

def dump_pickle(fname,data):
    with open(fname,'wb') as f:
        dump(data, f)
        
def load_pickle(fname):
    with open(fname,'rb') as f:
        return load(f)

def dump_blosc(fname,data):
    arr = dumps(data, -1)
    with open(fname, "wb") as f:
        s = 0
        while s < len(arr):
            e = min(s + blosc.MAX_BUFFERSIZE, len(arr))
            carr = blosc.compress(arr[s:e], typesize=8)
            f.write(carr)
            s = e
        
def load_blosc(fname):
    arr = []
    buffsize = blosc.MAX_BUFFERSIZE
    with open(fname, "rb") as f:
        while buffsize > 0:
            try:
                carr = f.read(buffsize)
            except (OverflowError, MemoryError):
                buffsize = buffsize // 2
                continue

            if len(carr) == 0:
                break
            arr.append(blosc.decompress(carr))
    return loads(b"".join(arr))