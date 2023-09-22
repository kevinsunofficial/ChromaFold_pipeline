import os
import os.path as osp
import numpy as np
import pandas as pd
import scipy
import scipy.stats
import warnings

warnings.filterwarnings('ignore')


def topdom(pred_mat, window_size=10):
    if pred_mat.shape[0]-pred_mat.shape[1]:
        raise ValueError(
            'Dimension mismatch ({}, {})'.format(pred_mat.shape[0], pred_mat.shape[1])
        )
    pad_mat = np.pad(pred_mat, window_size, mode='constant', constant_values=np.nan)
    dim = pad_mat.shape[0]
    signal = np.array([
        np.nanmean(pad_mat[i-window_size:i, i:i+window_size]) for i in range(dim)
    ][window_size+1: -window_size])

    return signal


def interpolate(signal, bin_size=10000, smooth=False):
    if smooth:
        l = len(signal) * bin_size
        sparse, compact = np.linspace(0, l, len(signal)), np.linspace(0, l, l)
        interp_signal = np.interp(compact, sparse, signal)
    else:
        interp_signal = np.tile(signal, (bin_size, 1)).flatten('F')
    
    return interp_signal


def similarity(signal1, signal2, window_size=100):
    if len(signal1)-len(signal2):
        raise ValueError(
            'Different signal1.length and signal2.length'
        )
    l = len(signal1)
    score = np.array([
        scipy.stats.pearsonr(
            signal1[i:i+window_size], signal2[i:i+window_size]
        )[0] for i in range(l-window_size)
    ])

    return score


if __name__=='__main__':
    pass