import os
import os.path as osp
import numpy as np
import pandas as pd
import scipy
import scipy.stats
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


def topdom(pred_mat, window_size=10, cutoff=None):
    if pred_mat.shape[0]-pred_mat.shape[1]:
        raise ValueError(
            'Dimension mismatch ({}, {})'.format(pred_mat.shape[0], pred_mat.shape[1])
        )
    pad_mat = np.pad(pred_mat, window_size, mode='constant', constant_values=np.nan)
    dim = pad_mat.shape[0]
    signal = np.array([
        np.nanmean(pad_mat[i-window_size:i, i:i+window_size]) for i in range(dim)
    ][window_size+1: -window_size])
    if cutoff is not None:
        signal[signal<cutoff] = cutoff

    return signal


def interpolate(signal, bin_size=10000, pattern='smooth'):
    if pattern is None: return signal
    if pattern not in ['smooth', 'zigzag']:
        raise ValueError(
            'Bad parameter, expecting \'smooth\' or \'zigzag\' but got \'{}\''.format(pattern)
        )
    if pattern=='smooth':
        l = len(signal) * bin_size
        sparse, compact = np.linspace(0, l, len(signal)), np.linspace(0, l, l)
        interp_signal = np.interp(compact, sparse, signal)
    else:
        interp_signal = np.tile(signal, (bin_size, 1)).flatten('F')
    
    return interp_signal


def similarity(signal1, signal2, window_size=100):
    if len(signal1)-len(signal2):
        raise ValueError(
            'Different signal1.length ({}) and signal2.length ({})'.format(len(signal1), len(signal2))
        )
    l = len(signal1)
    score = np.array([
        scipy.stats.pearsonr(
            signal1[i:i+window_size], signal2[i:i+window_size]
        )[0] for i in range(l-window_size)
    ])
    score[np.isnan(score)] = 1

    return score


def threshold(score, cutoff=0.7, margin=10000):
    indices = np.argwhere(score <= cutoff).flatten()
    starts, ends = [], []
    s, e = 0, 0
    for i in tqdm(indices, desc='selecting significant regions', position=0, leave=True):
        if not s and not e: s, e = i, i
        else:
            if i - e <= margin: e = i
            else:
                starts.append(s)
                ends.append(e)
                s, e = i, i
    if e != ends[-1]:
        starts.append(s)
        ends.append(e)
    regions = pd.DataFrame({
        'start': np.array(starts) - margin,
        'end': np.array(ends) + margin
    })
    return regions


if __name__=='__main__':
    pass