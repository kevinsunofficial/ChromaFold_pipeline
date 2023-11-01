import os
import os.path as osp
import numpy as np
import pandas as pd
import scipy
import scipy.stats
from scipy.signal import find_peaks
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


def quantile_norm(pred1, pred2):
    if pred1.shape[0] - pred1.shape[1]:
        raise ValueError(
            'Matrix 1 is not square ({}, {})'.format(pred1.shape[0], pred1.shape[1])
        )
    if pred2.shape[0] - pred2.shape[1]:
        raise ValueError(
            'Matrix 2 is not square ({}, {})'.format(pred2.shape[0], pred2.shape[1])
        )
    if pred1.shape[0] - pred2.shape[0]:
        raise ValueError(
            'Matrix dimension mismatch ({} vs {})'.format(pred1.shape[0], pred2.shape[0])
        )
    l = pred1.shape[0]
    pred1_diag = np.array([
        np.pad(np.diagonal(pred1, offset=i), (0, i), 'constant') for i in range(200)
    ]).T
    pred2_diag = np.array([
        np.pad(np.diagonal(pred2, offset=i), (0, i), 'constant') for i in range(200)
    ]).T
    pred = np.column_stack((pred1_diag.ravel(), pred2_diag.ravel()))
    df, df_sort = pd.DataFrame(pred), pd.DataFrame(np.sort(pred, axis=0))
    df_mean = df_sort.mean(axis=1)
    df_mean.index += 1
    df_qn = df.rank(method='min').stack().astype(int).map(df_mean).unstack()
    pred1_stripe, pred2_stripe = df_qn[0].values.reshape(-1, 200), df_qn[1].values.reshape(-1, 200)

    pred1_qn, pred2_qn = np.zeros_like(pred1), np.zeros_like(pred2)
    for i in range(200):
        idx = np.arange(l - i, dtype=int)
        pred1_qn[idx, idx+i] = pred1_qn[idx+i, idx] = pred1_stripe[:l-i, i]
        pred2_qn[idx, idx+i] = pred2_qn[idx+i, idx] = pred2_stripe[:l-i, i]
    
    return pred1_qn, pred2_qn


def topdom(pred_mat, window_size=10, cutoff=None):
    if pred_mat.shape[0]-pred_mat.shape[1]:
        raise ValueError(
            'Matrix is not square ({}, {})'.format(pred_mat.shape[0], pred_mat.shape[1])
        )
    pad_mat = np.pad(pred_mat, window_size, mode='constant', constant_values=np.nan)
    dim = pad_mat.shape[0]
    signal = np.array([
        np.nanmean(pad_mat[i-window_size:i, i:i+window_size]) for i in range(dim)
    ][window_size+1: -window_size])
    if cutoff is not None:
        signal[signal<cutoff] = cutoff

    return signal


def region_topdom(pred_mat, window_size=10):
    if pred_mat.shape[0]-pred_mat.shape[1]:
        raise ValueError(
            'Matrix is not square ({}, {})'.format(pred_mat.shape[0], pred_mat.shape[1])
        )
    pad_mat = np.pad(pred_mat, window_size, mode='constant', constant_values=np.nan)
    dim = pad_mat.shape[0]
    signal = np.array([
        np.nanmean(pad_mat[i-window_size:i+window_size, i-window_size:i+window_size]) for i in range(dim)
    ][window_size:-window_size])

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


def get_tads(mat, sizes, prominence=0.25):
    signal = np.array([region_topdom(mat, i) for i in sizes])
    rows, idxs = [], []
    for i in range(len(signal)):
        idx = find_peaks(signal[i], prominence=(prominence,))[0]
        row = np.full_like(idx, i)
        rows.append(row)
        idxs.append(idx)
    tads = np.array([
        np.concatenate(rows, axis=None), np.concatenate(idxs, axis=None)
    ])

    return tads


def tads_to_coords(tads, sizes):
    coords = np.array([
        tads[1] - sizes[tads[0]], tads[1] + sizes[tads[0]]
    ])

    return coords


def merge_coords(coords, sizes, close=5):
    df = pd.DataFrame({'x': coords[0], 'y': coords[1]})
    merged = df.groupby('x', as_index=False).agg({'y': 'max'})
    merged['s'] = (merged.y - merged.x) // 2
    merged = merged.sort_values(by=['x']).reset_index(drop=True)

    i = 0
    curx, cury, curs = merged.iloc[i]

    while i+1 < merged.shape[0]:
        x, y, s = merged.iloc[i+1]
        if s == curs:
            if abs(x - curx) <= close:
                curx, cury = min(curx, x), max(cury, y)
                curs = (cury - curx) // 2
                merged = merged.drop(i+1, axis=0).reset_index(drop=True)
            else:
                curx, cury, curs = x, y, s
                i += 1
        else:
            if abs(x - curx) <= close or abs(y - cury) <= close:
                curx, cury = min(curx, x), max(cury, y)
                curs = (cury - curx) // 2
                merged = merged.drop(i+1, axis=0).reset_index(drop=True)
            else:
                curx, cury, curs = x, y, s
                i += 1
        merged.iloc[i] = [curx, cury, curs]
    
    return merged.values.T


def get_tad_coords(pred1, pred2=None, min_dim=10, max_dim=100, num_dim=10, close=5):

    def generate_sizes(min_dim, max_dim, num_dim):
        min_dim, max_dim = max(1, min_dim), min(100, max_dim)
        return np.linspace(min_dim, max_dim, num=num_dim, dtype=int)
    
    sizes = generate_sizes(min_dim, max_dim, num_dim)
    tads1 = get_tads(pred1, sizes)
    if pred2 is not None:
        tads2 = get_tads(pred2, sizes)
        alltads = np.concatenate((tads1, tads2), axis=1)
        allcoords = tads_to_coords(alltads, sizes)
    else:
        allcoords = tads_to_coords(tads1, sizes)
    coords = merge_coords(allcoords, sizes, close)

    return coords


def rank_coords(pred1, pred2, coords):
    xs, ys, ss, diff_dirs, abs_scores = [], [], [], [], []
    for i in range(coords.shape[1]):
        x, y, s = coords[:, i]
        area1, area2 = pred1[x:y+1, x:y+1], pred2[x:y+1, x:y+1]
        area_diff = area1 - area2
        diff_dir = np.mean(area_diff)
        abs_score = np.std(area_diff) * np.ptp(area_diff)
        xs.append(x)
        ys.append(y)
        ss.append(s)
        diff_dirs.append(diff_dir)
        abs_scores.append(abs_score)
    df = pd.DataFrame({
        'x_coord': xs, 'y_coord': ys, 'window_size': ss,
        'diff_direction': diff_dirs, 'abs_diff_score': abs_scores
    })
    ranked = df.sort_values(by=['abs_diff_score'], ignore_index=True, ascending=False)

    return ranked


def threshold(score, cutoff=0.7, kernel='diff', margin=1000):
    if kernel == 'diff':
        indices = np.argwhere(np.abs(score) >= cutoff).flatten()
    elif kernel == 'pearson':
        indices = np.argwhere(score <= cutoff).flatten()
    elif kernel == 'tad_diff':
        indices = np.argwhere(score >= cutoff).flatten()
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
    if not ends:
        if s and e:
            starts.append(s)
            ends.append(e)
    else:
        if e != ends[-1]:
            starts.append(s)
            ends.append(e)
    regions = None
    if starts and ends:
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


def verification(pred, start, window_size, cutoff=0):
    if pred.shape[0]-pred.shape[1]:
        raise ValueError(
            'Dimension mismatch ({}, {})'.format(pred.shape[0], pred.shape[1])
        )
    mat = pred[start:start+700, start:start+700]
    score = []
    for i in range(300, 500):
        val = np.nanmean(mat[i-window_size:i, i:i+window_size])
        if cutoff is not None:
            val = cutoff if val < cutoff else val
        score.append(val)
    normal = gaussian(len(score), window_size)
    normalscore = np.array(score).dot(normal)

    return normalscore


def verification_paired(pred1, pred2, start, window_size, cutoff=0):
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


def match_tad_score(ranked, start):
    this_diff_dir, this_abs_score = 0, 0
    for i in range(ranked.shape[0]):
        x, y, _, diff_dir, abs_score = ranked.iloc[i]
        if x <= start <= y:
            this_diff_dir, this_abs_score = diff_dir, abs_score
            return this_diff_dir, this_abs_score
    
    return this_diff_dir, this_abs_score


if __name__=='__main__':
    pass