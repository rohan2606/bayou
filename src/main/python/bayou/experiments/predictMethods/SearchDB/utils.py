import numpy as np

def rank_statistic(_rank, total, prev_hits, cutoff):
    cutoff = np.array(cutoff)
    hits = prev_hits + (_rank < cutoff)
    prctg = hits / total
    return hits, prctg

def ListToFormattedString(alist, Type):
    # Each item is right-adjusted, width=3
    if Type == 'float':
        formatted_list = ['{:.2f}' for item in alist]
        s = ','.join(formatted_list)
    elif Type == 'int':
        formatted_list = ['{:>3}' for item in alist]
        s = ','.join(formatted_list)
    return s.format(*alist)
