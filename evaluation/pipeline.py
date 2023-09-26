import subprocess
import os
import os.path as osp
import numpy as np
import pandas as pd
import scipy
import scipy.stats
import gffutils
import argparse
import sqlite3
import json
from dataloader import *
from metrics import *

import warnings
warnings.filterwarnings('ignore')


def pipe_single(args):
    pred_dir = args.pred_dir
    ct = args.ct[0]
    chrom = args.chrom
    pred_len = args.pred_len
    avg_stripe = args.avg_stripe
    topdom_window_size = args.topdom_w
    topdom_cutoff = args.topdom_cutoff

    pred_mat = load_pred(pred_dir, ct, chrom, pred_len=pred_len, avg_stripe=avg_stripe)
    signal = topdom(pred_mat, window_size=topdom_window_size, cutoff=topdom_cutoff)

    return signal


def pipe_pair(args):
    pred_dir = args.pred_dir
    ct1, ct2 = args.ct[:2]
    chrom = args.chrom
    pred_len = args.pred_len
    avg_stripe = args.avg_stripe
    topdom_window_size = args.topdom_window
    topdom_cutoff = args.topdom_cutoff
    similar_window_size = args.similar_window
    bin_size = args.bin_size
    pattern = args.pattern
    thresh_cutoff = args.thresh_cutoff
    thresh_margin = args.thresh_margin
    db_file = args.db_file
    gtf_file = args.gtf_file
    table = args.table
    featuretype = args.featuretype
    filters = args.filters

    print('Loading predictions...')
    pred1 = load_pred(pred_dir, ct1, chrom, pred_len=pred_len, avg_stripe=avg_stripe)
    pred2 = load_pred(pred_dir, ct2, chrom, pred_len=pred_len, avg_stripe=avg_stripe)
    print('Calculating TopDom insulation score...')
    signal1 = topdom(pred1, window_size=topdom_window_size, cutoff=topdom_cutoff)
    signal2 = topdom(pred2, window_size=topdom_window_size, cutoff=topdom_cutoff)
    print('Calculating Pearson Correlations...')
    raw_simscore = similarity(signal1, signal2, window_size=similar_window_size)
    simscore = interpolate(raw_simscore, bin_size=bin_size, pattern=pattern)
    print('Selecting significant regions...')
    regions = threshold(simscore, cutoff=thresh_cutoff, margin=thresh_margin)
    queries = generate_query(regions, chrom=chrom, table=table, featuretype=featuretype)
    print('Querying databse...')
    db = load_database(db_file, gtf_file)
    res = db_query(db, queries, filters=filters)

    if res is not None:
        res.to_csv(osp.join(out_dir, 'chr{}_significant_genes.csv'.format(chrom)), header=True, index=False)

    return res


if __name__=='__main__':
    os.system('clear')

    print('Parsing arguments...')

    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', required=True, type=str, help='ChromaFold prediction result directory')
    parser.add_argument('--paired', required=False, action='store_true', default=False, help='Indicate whether the analysis is for paired prediction')
    parser.add_argument('--ct', required=True, nargs='+', default=[], help='Full cell type names, for paired this would be two cell types')
    parser.add_argument('--chrom', required=True, type=int, help='Chromosome number')
    parser.add_argument('--pred_len', required=False, type=int, default=200, help='Prediction length, default=200')
    parser.add_argument('--avg_stripe', required=False, action='store_true', help='Average V-stripe, default=False')

    parser.add_argument('--topdom_window', required=False, type=int, default=10, help='Window size for running TopDom, default=10')
    parser.add_argument('--topdom_cutoff', required=False, type=float, default=None, help='Cutoff for running TopDom, anything below will be set to cutoff, default=None')

    parser.add_argument('--similar_window', required=False, type=int, default=100, help='Window size for running sliding window Pearson Correlation, default=100')

    parser.add_argument('--bin_size', required=False, type=int, default=10000, help='Bin size when running ChromaFold, default=10kb=10000')
    parser.add_argument('--pattern', required=False, type=str, default=None, help='Filling behavior for TopDom score interpolation, default=None')

    parser.add_argument('--thresh_cutoff', required=False, type=float, default=0.6, help='Cutoff for selecting window with difference, default=0.6')
    parser.add_argument('--thresh_margin', required=False, type=int, default=10000, help='Margin of error used when extending window with difference, default=10000')

    parser.add_argument('--db_file', required=True, type=str, help='Database file directory')
    parser.add_argument('--gtf_file', required=False, type=str, default='gencode.vM10.basic.annotation.gtf',help='GTF file directory')
    parser.add_argument('--table', required=False, type=str, default='features', help='Table name for db query')
    parser.add_argument('--featuretype', required=False, type=str, default='gene', help='Feature types to select for db query')
    parser.add_argument('--filters', required=False, nargs='+', default=[], help='Attribute filters in database query, input each filter with \"key=value\" format')

    parser.add_argument('--out_dir', required=True, type=str, help='Output directory to store the result')

    args = parser.parse_args()

    paired = args.paired
    out_dir = args.out_dir
    
    if paired:
        res = pipe_pair(args)
    else:
        res = pipe_single(args)
    
    print('DONE')
    

