import os
import os.path as osp
import numpy as np
import pandas as pd
import scipy
import torch
from scipy.sparse import csr_matrix
from scipy.signal import find_peaks
from tqdm import tqdm
import warnings
import sqlite3
import json

warnings.filterwarnings('ignore')


def quantile_normalize(preds, offset=2):
    N, H, W = preds.shape
    assert H == W, f'Matrix is not square ({H}, {W})'
    pred_diag = np.column_stack((
        np.concatenate([
            np.diag(pred, i) for i in range(offset, 200)
        ]) for pred in preds
    ))
    df, df_mean = pd.DataFrame(pred_diag), pd.DataFrame(np.sort(pred_diag, axis=0)).mean(axis=1)
    df_mean.index += 1
    pred_diag_qn = df.rank(method='min').stack().astype(int).map(df_mean).unstack().values
    pred_diag_qn = pred_diag_qn.T.reshape(N, -1)
    preds_qn = np.zeros_like(preds)
    current = 0
    for i in range(offset, 200):
        idx = np.arange(H - i, dtype=int)
        l = np.diagonal(preds_qn, offset=i, axis1=1, axis2=2).shape[1]
        preds_qn[:, idx, idx+i] = preds_qn[:, idx+i, idx] = pred_diag_qn[:, current:current+l]
        current += l
    
    return preds_qn


def topdom(matrix, window_size, cutoff=None):
    assert matrix.shape[0] == matrix.shape[1], f'Matrix is not square {matrix.shape}'
    pad_mat = np.pad(matrix, window_size, mode='constant', constant_values=np.nan)
    dim = pad_mat.shape[0]
    signal = np.array([
        np.nanmean(pad_mat[i-window_size:i, i:i+window_size]) for i in range(dim)
    ][window_size+1:-window_size])
    if cutoff is not None:
        signal[signal < cutoff] = cutoff
    
    return signal


def get_tad_vertex(pred, min_dim=10, max_dim=90, num_dim=25, close=15):
    sizes = np.linspace(max(1, min_dim), min(max_dim, 100), num=num_dim, dtype=int)
    x, y = [], []
    for i, s in enumerate(tqdm(sizes)):
        signal = topdom(pred, s)
        peak = find_peaks(signal, prominence=0.25, height=0.1)[0]
        trough = find_peaks(-signal, prominence=0.25, height=0.1)[0]
        vertex = np.concatenate((peak, trough), axis=None)
        x.extend((vertex - s).tolist())
        y.extend((vertex + s).tolist())
    
    raw_vertex = pd.DataFrame({'x': x, 'y': y}).sort_values(['x', 'y']).reset_index(drop=True)
    i, current = 0, None
    merged = []

    while i < raw_vertex.shape[0]:
        if not merged:
            merged.append(raw_vertex.iloc[i].values)
        else:
            current = raw_vertex.iloc[i].values
            diff = np.abs(merged[-1] - current)
            if np.sum(diff) <= close:
                merged[-1] = np.array([
                    min(merged[-1][0], current[0]),
                    max(merged[-1][1], current[1])
                ])
            else:
                merged.append(current[:])
        i += 1

    merged = np.array(merged)
    merged_vertex = pd.DataFrame({'start': merged[:, 0], 'end': merged[:, 1]})

    return merged_vertex


def generate_query(chrom, start=None, end=None, length=None, featuretype=None):
    clause = ['SELECT seqid, start, end, attributes FROM features']
    where_clause = [f'WHERE seqid = "chr{chrom}"']
    if start is not None:
        where_clause.append(f'start >= {start}')
    if end is not None:
        where_clause.append(f'end <= {end}')
    if length is not None:
        where_clause.append(f'end - start >= {length}')
    if featuretype is not None:
        where_clause.append(f'featuretype = "{featuretype}"')
    clause.append(' AND '.join(where_clause))
    query = ' '.join(clause)
    
    return query
    

def check_attr(attr, filters=[]):
    for restr in filters:
        assert '=' in restr, f'Cannot parse filter {restr}. Please doublecheck the input'
        k, v = restr.strip().split('=')
        k, v = k.strip(), v.strip()
        assert k in attr, f'Cannot find key {k} in attributes. Please doublecheck the input'
        if attr[k][0] != v:
            return False
    
    return True


def db_query(db, chrom, start=None, end=None, length=10000, featuretype='gene', 
             filters=['gene_type=protein_coding']):
    query = generate_query(chrom, start, end, length, featuretype)
    itr = db.execute(query).fetchall()

    seqid_, start_, end_, gene_name_, gene_id_ = [], [], [], [], []
    valid = 0
    for obj in itr:
        attr = json.loads(obj['attributes'])
        if not check_attr(attr, filters):
            continue
        seqid_.append(obj['seqid'])
        start_.append(obj['start'])
        end_.append(obj['end'])
        gene_name_.append(attr['gene_name'][0])
        gene_id_.append(attr['gene_id'][0])
        valid += 1

    if valid:
        info = pd.DataFrame({
            'chrom': seqid_, 'start': start_, 'end': end_, 
            'gene_name': gene_name_, 'gene_id': gene_id_
        })
        print(f'Query completed with {valid} match(es)')
    else:
        warnings.warn(
            'No match found. Please consider changing the filters',
            RuntimeWarning
        )

    return valid, info


def create_bedpe(pred, chrom, ct, bedpe_dir, bedpe_thresh=99.):
    assert pred.shape[0] == pred.shape[1], f'Matrix is not square {pred.shape}'
    min_len = len(np.diag(pred, 199))
    all_zval = np.concatenate([np.diag(pred, i)[:min_len] for i in range(2, 200)])
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
    coords = np.column_stack((coords[:, 0], coords[:, 0], coords[:, 1], coords[:, 1], coords[:, 2]))

    bedpe = []

    for s1, e1, s2, e2, score in coords:
        line = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
            chrom, int(s1 * 1e4), int(e1 * 1e4), 
            chrom, int(s2 * 1e4), int(e2 * 1e4),
            f'chr{chrom}_gene', score, '.', '.'
        )
        bedpe.append(line)
    
    bedpe_file = osp.join(bedpe_dir, f'{ct}_chr{chrom}.bedpe')
    with open(bedpe_file, 'w') as file:
        file.writelines(bedpe)
