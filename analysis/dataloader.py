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


def load_ctcf(ctcf_path, chrom, start=None):
    ctcf_all = pickle.load(open(ctcf_path, 'rb'))
    ctcf = ctcf_all['chr{}'.format(chrom)].toarray()[0]
    if start is not None:
        ctcf = ctcf[start*200: (start+700)*200]

    return ctcf


def load_atac(atac_path, chrom, start=None):
    atac_all = pickle.load(open(atac_path, 'rb'))
    atac = atac_all['chr{}'.format(chrom)].flatten()
    if start is not None:
        atac = atac[start*200: (start+700)*200]

    return atac


def process_scatac(scatac_pre, metacell, start):
    tmp = torch.tensor((metacell * scatac_pre)[:, start*20:(start+700)*20].toarray()).T

    size, eps = tmp.shape[1], 1e-8
    one, zero = torch.tensor(1.0), torch.tensor(0.0)
    lrg = torch.where(tmp>0, one, zero)
    eql = torch.where(tmp==0, one, zero)
    num, denom = lrg @ lrg.T, size - eql @ eql.T
    scatac = torch.div(num, torch.max(denom, eps * torch.ones_like(denom)))
    scatac[scatac != scatac] = 0

    scatac = scatac.reshape(
        scatac.shape[0]//20, 20, -1
    ).mean(axis=1).reshape(
        -1, scatac.shape[1]//20, 20
    ).mean(axis=2)

    return scatac


def load_scatac(scatac_path, metacell_path, chrom, start=None):
    scatac_pre = pickle.load(open(scatac_path, 'rb'))['chr{}'.format(chrom)]
    metacell = csr_matrix(pd.read_csv(metacell_path, index_col=0).values)

    if start is not None:
        scatac = process_scatac(scatac_pre, metacell, start)
        return scatac, metacell
    else:
        return scatac_pre, metacell


def load_multiome(input_dir, ct, chrom, start=None, genome='mm10'):
    ctcf_path = osp.join(input_dir, 'dna', '{}_ctcf_motif_score.p'.format(genome))
    atac_path = osp.join(input_dir, 'atac', '{}_tile_pbulk_50bp_dict.p'.format(ct))
    scatac_path = osp.join(input_dir, 'atac', '{}_tile_500bp_dict.p'.format(ct))
    metacell_path = osp.join(input_dir, 'atac', '{}_metacell_mask.csv'.format(ct))

    ctcf = load_ctcf(ctcf_path, chrom, start=start)
    atac = load_atac(atac_path, chrom, start=start)
    scatac, metacell = load_scatac(scatac_path, metacell_path, chrom, start=start)

    return ctcf, atac, scatac, metacell


def set_diagonal(mat, value=0):
    if mat.shape[0] - mat.shape[1]:
        raise ValueError(
            'Matrix is not square ({}, {})'.format(mat.shape[0], mat.shape[1])
        )
    l = mat.shape[0]
    idx = np.arange(l)
    mat[idx[:-1], idx[1:]], mat[idx[1:], idx[:-1]] = value, value

    return mat


def load_pred(pred_dir, ct, chrom, pred_len=200, avg_stripe=True):
    file = osp.join(pred_dir, ct, 'prediction_{}_chr{}.npz'.format(ct, chrom))
    temp = np.load(file)['arr_0']
    chrom_len = temp.shape[0]
    prep = np.insert(temp, pred_len, 0, axis=1)
    mat = np.array([
        np.insert(np.zeros(chrom_len+pred_len+1), i, prep[i]) for i in range(chrom_len)
    ])
    summed = np.vstack((
        np.zeros((pred_len, mat.shape[1])), mat
    )).T[:chrom_len+pred_len, :chrom_len+pred_len]
    if avg_stripe:
        summed = (summed + np.vstack((
            np.zeros((pred_len, mat.shape[1])), mat
        ))[:chrom_len+pred_len, :chrom_len+pred_len])/2
    
    pred = set_diagonal(summed[pred_len:-pred_len, pred_len:-pred_len])

    return pred


def load_database(db_file, gtf_file):
    if osp.isfile(db_file):
        db = gffutils.FeatureDB(db_file)
    else:
        print('creating db from raw. This might take a while.')
        db = gffutils.create_db(gtf_file, db_file)
    
    return db


def parse_query(line):
    start, end, chrom, table, featuretype = line
    select_from = 'SELECT * FROM {} WHERE'.format(table)
    start_req = 'start >= {}'.format(start)
    end_req = 'end <= {}'.format(end)
    len_req = 'end - start >= 1000'
    chrom_req = 'seqid = \"chr{}\"'.format(chrom)
    feature_req = 'featuretype = \"{}\"'.format(featuretype)
    where_reqs = ' AND '.join([start_req, end_req, len_req, chrom_req, feature_req])
    query = ' '.join([select_from, where_reqs])
    
    return query


def parse_query_tad(line):
    _, _, _, _, _, left, right, chrom, table, featuretype = line
    select_from = f'SELECT * FROM {table} WHERE'
    start_req = f'start >= {left} AND start <= {right}'
    end_req = f'end >= {left} AND end <= {right}'
    len_req = 'end - start >= 1000'
    chrom_req = f'seqid = \"chr{chrom}\"'
    feature_req = f'featuretype = \"{featuretype}\"'
    where_reqs = ' AND '.join([start_req, end_req, len_req, chrom_req, feature_req])
    query = ' '.join([select_from, where_reqs])
    
    return query


def generate_query(regions, chrom, table='features', featuretype='gene'):
    regions['chrom'] = chrom
    regions['table'] = table
    regions['featuretype'] = featuretype
    reqs = regions.apply(parse_query, 1)

    return reqs


def generate_query_tad(ranked, chrom, table='features', featuretype='gene'):
    ranked['left'] = (ranked.x_coord - 10) * int(1e4)
    ranked['right'] = (ranked.y_coord + 10) * int(1e4)
    ranked['chrom'] = chrom
    ranked['table'] = table
    ranked['featuretype'] = featuretype
    ranked['reqs'] = ranked.apply(parse_query_tad, 1)

    return ranked


def merge_attr(attrs):
    if not attrs:
        raise ValueError(
            'No attributes detected. Please consider expanding your search by changing the filters'
        )
    return {k: [attr[k][0] if k in attr else None for attr in attrs] for k in attrs[0].keys()}


def check_attr(attr, filters):
    for restr in filters:
        if '=' not in restr:
            raise ValueError(
                'Cannot parse filter {}. Please doublecheck the input'.format(restr)
            )
        k, v = restr.strip().split('=')
        k, v = k.strip(), v.strip()
        if k not in attr:
            raise ValueError(
                'Cannot find key {} in attributes. Please doublecheck the input'.format(k)
            )
        if attr[k][0] != v: return False
    
    return True


def db_query_tad(db, ranked, chrom, table='features', featuretype='gene', filters=[]):
    ranked = generate_query_tad(ranked, chrom=chrom, table=table, featuretype=featuretype)
    select = [
        'chrom', 'start', 'end', 'gene_name', 'gene_id',
        'gene_type', 'level', 'score', 'abs_score'
    ]
    all_df = []
    for i in range(ranked.shape[0]):
        _, _, _, score, abs_score, _, _, chrom, table, featuretype, query = ranked.iloc[i]
        chrom, start, end, attrs = [], [], [], []
        valid = 0
        itr = db.execute(query).fetchall()
        for obj in itr:
            attr = json.loads(obj['attributes'])
            if filters:
                if not check_attr(attr, filters): continue
            chrom.append(obj['seqid'])
            start.append(obj['start'])
            end.append(obj['end'])
            attrs.append(attr)
            valid += 1
        if valid:
            info = pd.DataFrame({
                'chrom': chrom, 'start': start, 'end': end
            })
            attrs = pd.DataFrame(merge_attr(attrs))
            res = pd.concat([info, attrs], axis=1)
            res['score'] = score
            res['abs_score'] = abs_score
            res = res[select]
            all_df.append(res)
    if all_df:
        res = pd.concat(all_df, axis=0)
        res = res.drop_duplicates(subset='gene_id', keep='first', ignore_index=True)
        valid = res.shape[0]
        print(f'{ranked.shape[0]} databse query completed with {valid} match(es)')
    else:
        res, valid = None, 0
        warnings.warn(
            'No match found. Please consider expanding your search by changing the filters',
            RuntimeWarning
        )
    
    return res, valid        


def parse_res(row):
    start = int(max(row['start']//1e4 - 400, 0))
    locstart, locend = row['start']/1e4 - start, row['end']/1e4 - start
    gene = row['gene_name']

    return start, locstart, locend, gene
    

if __name__=='__main__':
    pass
