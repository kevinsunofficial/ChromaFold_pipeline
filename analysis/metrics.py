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


def quantile_normalize(preds, offset=2):
    N, H, W = preds.shape
    assert H == W, f'Matrix is not square ({H}, {W})'
    pred_diag = np.column_stack((
        np.array([
            np.pad(np.diagonal(pred, offset=i), (0, i), 'constant') for i in range(offset, 200)
        ]).T.ravel() for pred in preds
    ))
    df, df_mean = pd.DataFrame(pred_diag), pd.DataFrame(np.sort(pred_diag, axis=0)).mean(axis=1)
    df_mean.index += 1
    pred_diag_qn = df.rank(method='min').stack().astype(int).map(df_mean).unstack().values
    pred_diag_qn = pred_diag_qn.T.reshape(N, -1, 200-offset)
    preds_qn = np.zeros_like(preds)
    for i in range(offset, 200):
        idx = np.arange(H - i, dtype=int)
        preds_qn[:, idx, idx+i] = preds_qn[:, idx+i, idx] = pred_diag_qn[:, :H-i, i-offset]
    
    return preds_qn


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


def score_genes(pred, res):
    l = pred.shape[0]
    scores, abs_scores = [], []

    for i in range(res.shape[0]):
        start, end = res.iloc[i].start, res.iloc[i].end
        start, end = start // int(1e4), end // int(1e4)
        stripe = []
        for i in range(max(200, start - 10), min(end + 11, l - 200)):
            left, right = pred[i-200:i, i], pred[i, i+1:i+201]
            left_pad = np.pad(left, (200 - len(left), 0), 'empty')
            right_pad = np.pad(right, (0, 200 - len(right)), 'empty')
            stripe.append(np.array([left_pad, right_pad]).flatten())
        stripe = np.array(stripe)
        stripe[(stripe < 1) & (stripe > -1)] = 0
        scores.append(np.nansum(stripe))
        abs_scores.append(np.nansum(np.abs(stripe)))
    
    res['score'] = scores
    res['abs_score'] = abs_scores
    res = res.sort_values(by='abs_score', ignore_index=True, ascending=False)

    return res


def merge_bedpe(coords, bedpe_margin):
    s1, e1, s2, e2 = 0, 0, 0, 0
    sumscore, count = 0, 0
    rows = []

    for c1, c2, score in coords:
        if not count:
            s1, e1, s2, e2 = c1, c1, c2, c2
            sumscore, count = score, 1
        else:
            if not (s1 <= c1 <= e1+bedpe_margin and s2 <= c2 <= e2+bedpe_margin):
                rows.append(np.array([s1, e1, s2, e2, sumscore]))
                s1, e1, s2, e2 = c1, c1, c2, c2
                sumscore, count = score, 1
            else:
                sumscore += 1
                count += 1
                e1, e2 = max(e1, c1), max(e2, c2)
    if count:
        rows.append(np.array([s1, e1, s2, e2, sumscore]))
    
    return np.concatenate(rows).reshape(-1, 5)


def parse_bedpe(pred, bedpe_thresh=99., bedpe_margin=None):
    if pred.shape[0]-pred.shape[1]:
        raise ValueError(
            'Dimension mismatch ({}, {})'.format(pred.shape[0], pred.shape[1])
        )
    min_len = len(np.diag(pred, 199))
    all_zval = np.concatenate([np.diag(pred, i)[:min_len] for i in range(1, 200)])
    bin_pred = all_zval.reshape(-1, min_len)
    bin_mask = (bin_pred.sum(0) < np.percentile(bin_pred.sum(0), 1))
    zval_cutoff = np.percentile(all_zval, bedpe_thresh) if bedpe_thresh >= 90 else bedpe_thresh

    selected_pred = np.copy(bin_pred)
    selected_pred[:, bin_mask] = 0
    indices = np.argwhere(selected_pred > zval_cutoff)
    scores = selected_pred[selected_pred > zval_cutoff]

    coords = np.column_stack((indices[:, 1], np.array(indices[:, 1] + indices[:, 0] + 1), scores))
    coords = coords[np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))]
    coords = coords[np.diff(coords)[:, 0] >= 50, :]
    if bedpe_margin is not None:
        coords = merge_bedpe(coords, bedpe_margin=bedpe_margin)
    else:
        coords = np.column_stack((coords[:, 0], coords[:, 0], coords[:, 1], coords[:, 1], coords[:, 2]))

    return coords


if __name__=='__main__':
    pass