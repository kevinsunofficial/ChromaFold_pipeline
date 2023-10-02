import os
import os.path as osp
import numpy as np
import pandas as pd
import scipy
import scipy.stats
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


def topdom(pred_mat, window_size=10, cutoff=0):
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


def sim_pearson(signal1, signal2, window_size=10):
    l = len(signal1)
    score = np.array([
        scipy.stats.pearsonr(
            signal1[i:i+window_size], signal2[i:i+window_size]
        )[0] for i in range(l-window_size)
    ])
    score[score != score] = 1

    return score


def sim_difference(signal1, signal2):
    score = signal1 - signal2
    score[score != score] = 0

    return score


def similarity(signal1, signal2, kernel='diff', window_size=10):
    if len(signal1)-len(signal2):
        raise ValueError(
            'Different signal1.length ({}) and signal2.length ({})'.format(len(signal1), len(signal2))
        )
    if kernel == 'diff':
        score = sim_difference(signal1, signal2)
    elif kernel == 'pearson':
        score = sim_pearson(signal1, signal2, window_size=window_size)
    
    return score


def threshold(score, cutoff=0.7, kernel='diff', margin=1000):
    if kernel == 'diff':
        indices = np.argwhere(np.abs(score)>=cutoff).flatten()
    elif kernel == 'pearson':
        indices = np.argwhere(score <= cutoff).flatten()
    if len(indices) == 0:
        raise ValueError(
            'No valid result above threshold. Please consider expanding your search by changing the filters'
        )
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


def gaussian(n, sigma):
    kernel = np.array([
        1 / (n*np.sqrt(2*np.pi)) * np.exp(-i**2/(2*sigma**2)) for i in range(-n//2, n//2)
    ])
    kernel /= kernel.sum()

    return kernel


def verification(pred1, pred2, start, locstart, locend, window_size, cutoff=0):
    if pred1.shape[0]-pred1.shape[1]:
        raise ValueError(
            'Dimension mismatch ({}, {})'.format(pred1.shape[0], pred1.shape[1])
        )
    if pred2.shape[0]-pred2.shape[1]:
        raise ValueError(
            'Dimension mismatch ({}, {})'.format(pred2.shape[0], pred2.shape[1])
        )
    mat1, mat2 = pred1[start:start+700, start:start+700], pred2[start:start+700, start:start+700]
    score1, score2 = [], []
    for i in range(300, 500):
        val1 = np.nanmean(mat1[i-window_size:i, i:i+window_size])
        val2 = np.nanmean(mat2[i-window_size:i, i:i+window_size])
        if cutoff is not None:
            val1 = cutoff if val1 < cutoff else val1
            val2 = cutoff if val2 < cutoff else val2
        score1.append(val1)
        score2.append(val2)
    score = np.abs(np.array(score1) - np.array(score2))
    normal = gaussian(len(score), window_size)
    normalscore = score.dot(normal)

    return normalscore


if __name__=='__main__':
    pass