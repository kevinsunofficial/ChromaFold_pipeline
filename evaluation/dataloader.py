import os
import os.path as osp
import numpy as np
import pandas as pd
import gffutils
import argparse
import warnings

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
    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', required=True, type=str, help='ChromaFold prediction result directory')
    parser.add_argument('--ct', required=True, type=str, help='Full cell type name')
    parser.add_argument('--chrom', required=True, type=int, help='Chromosome number')
    parser.add_argument('--pred_len', required=False, type=int, default=200, help='Prediction length, default=200')
    parser.add_argument('--avg_stripe', required=False, action='store_true', help='Average V-stripe, default=False')
    args = parser.parse_args()

    pred_dir = args.pred_dir
    ct = args.ct
    chrom = args.chrom
    pred_len = args.pred_len
    avg_stripe = args.avg_stripe

    pred_mat = load_pred(pred_dir, ct, chrom, pred_len=pred_len, avg_stripe=avg_stripe)