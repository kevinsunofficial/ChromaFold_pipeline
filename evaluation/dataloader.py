import os
import os.path as osp
import numpy as np
import pandas as pd
import gffutils
import warnings
import sqlite3
import json

warnings.filterwarnings('ignore')


def load_pred(pred_dir, ct, chrom, pred_len=200, avg_stripe=False):
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
    
    return summed[pred_len:-pred_len, pred_len:-pred_len]


def load_database(db_file, gtf_file):
    if osp.isfile(db_file):
        db = gffutils.FeatureDB(db_file)
    else:
        db = gffutils.create_db(gtf_file, db_file)
    
    return db


def parse_query(line):
    start, end, chrom, table, featuretype = line
    select_from = 'SELECT * FROM {} WHERE'.format(table)
    start_req = 'start >= {}'.format(start)
    end_req = 'end <= {}'.format(end)
    chrom_req = 'seqid = \"chr{}\"'.format(chrom)
    feature_req = 'featuretype = \"{}\"'.format(featuretype)
    where_reqs = ' AND '.join([start_req, end_req, chrom_req, feature_req])
    query = ' '.join([select_from, where_reqs])
    
    return query


def generate_query(regions, chrom, table='features', featuretype='gene'):
    regions['chrom'] = chrom
    regions['table'] = table
    regions['featuretype'] = featuretype
    reqs = regions.apply(parse_query, 1)

    return reqs


def db_query(db, queries, restrictions=None):
    res = pd.DataFrame(columns=['gene_name', 'chrom', 'start', 'end'])
    for query in queries:
        itr = db.execute(query).fetchall()
        for obj in itr:
            attr = json.loads(obj['attributes'])
            if restrictions is not None:
                pass
            res.append(pd.DataFrame({
                'gene_name': attr['gene_name'][0],
                'chrom': obj['seqid'],
                'start': obj['start'],
                'end': obj['end']
            }))
    
    return res            
    

if __name__=='__main__':
    pass
