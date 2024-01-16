import os
import os.path as osp
import numpy as np
import pandas as pd
import scipy
import torch
from scipy.sparse import csr_matrix
import pickle
import gffutils
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
    