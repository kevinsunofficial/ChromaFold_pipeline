import os
import os.path as osp
import numpy as np
import pandas as pd
import scipy
import scipy.stats
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import gffutils
import argparse
import sqlite3
import json
from dataloader import *
from metrics import *
from plots import *

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


def sort_significance(args, res, valid, pred1, pred2):
    topdom_window_size = args.topdom_window
    topdom_cutoff = args.topdom_cutoff

    scores = []
    for i in range(valid):
        start, locstart, locend, _ = parse_res(res.iloc[i])
        score = verification(pred1, pred2, start, locstart, locend, window_size=topdom_window_size, cutoff=topdom_cutoff)
        scores.append(score)
    
    res['significance'] = scores
    res = res.sort_values(by=['significance'], ascending=False)

    return res


def plot_gene(args, data, start, locstart, locend, gene):
    fig_dir = args.fig_dir
    ct1, ct2 = args.ct[:2]

    ctcf, atac1, atac2, scatac_pre1, scatac_pre2, metacell1, metacell2, pred1, pred2 = data
    ctcf = ctcf[start*200: (start+700)*200]
    atac1, atac2 = atac1[start*200: (start+700)*200], atac2[start*200: (start+700)*200]
    scatac1 = process_scatac(scatac_pre1, metacell1, start)
    scatac2 = process_scatac(scatac_pre2, metacell2, start)

    savefig_dir = osp.join(fig_dir, 'chr{}'.format(chrom), 'chr{}_{}'.format(chrom, gene))
    if not osp.exists(savefig_dir):
        os.makedirs(savefig_dir)

    plot_ctcf(ctcf, chrom, start, gene, locstart, locend, savefig_dir)
    plot_atac(atac1, atac2, ct1, ct2, chrom, start, gene, locstart, locend, savefig_dir)
    plot_scatac(scatac1, scatac2, ct1, ct2, chrom, start, gene, locstart, locend, savefig_dir)
    plot_pred(pred1, pred2, ct1, ct2, chrom, start, gene, locstart, locend, savefig_dir)


def pairwise_difference(args):
    input_dir = args.input_dir
    pred_dir = args.pred_dir
    ct1, ct2 = args.ct[:2]
    chrom = args.chrom
    pred_len = args.pred_len
    avg_stripe = args.avg_stripe
    topdom_window_size = args.topdom_window
    topdom_cutoff = args.topdom_cutoff
    kernel = args.kernel
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
    numplot = args.num_plot
    out_dir = args.out_dir

    print('Loading multiome data...')
    ctcf, atac1, scatac1, metacell1 = load_multiome(input_dir, ct1, chrom, start=None)
    _, atac2, scatac2, metacell2 = load_multiome(input_dir, ct2, chrom, start=None)

    print('Loading predictions...')
    pred1 = load_pred(pred_dir, ct1, chrom, pred_len=pred_len, avg_stripe=avg_stripe)
    pred2 = load_pred(pred_dir, ct2, chrom, pred_len=pred_len, avg_stripe=avg_stripe)
    data = [ctcf, atac1, atac2, scatac1, scatac2, metacell1, metacell2, pred1, pred2]

    print('Calculating TopDom insulation score...')
    signal1 = topdom(pred1, window_size=topdom_window_size, cutoff=topdom_cutoff)
    signal2 = topdom(pred2, window_size=topdom_window_size, cutoff=topdom_cutoff)

    print('Calculating Similarity...')
    raw_simscore = similarity(signal1, signal2, kernel=kernel, window_size=similar_window_size)
    simscore = interpolate(raw_simscore, bin_size=bin_size, pattern=pattern)

    print('Selecting significant regions...')
    regions = threshold(simscore, cutoff=thresh_cutoff, margin=thresh_margin)
    queries = generate_query(regions, chrom=chrom, table=table, featuretype=featuretype)

    print('Querying database...')
    db = load_database(db_file, gtf_file)
    res, numvalid = db_query(db, queries, filters=filters)

    if numvalid:
        print('Sorting query result by significance')
        res = sort_significance(args, res, numvalid, pred1, pred2)
        res.to_csv(osp.join(out_dir, 'chr{}_significant_genes.csv'.format(chrom)), header=True, index=False)
        print('Plotting result...')
        if numplot is not None and numplot > 0:
            for i in tqdm(range(numplot), desc='plotting result', position=0, leave=True):
                row = res.iloc[i]
                start, locstart, locend, gene = parse_res(row)
                plot_gene(args, data, start, locstart, locend, gene)

    return res



if __name__=='__main__':

    print('\nParsing arguments...')
    os.system('clear')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, type=str, help='ChromaFold multiome input directory')
    parser.add_argument('--pred_dir', required=True, type=str, help='ChromaFold prediction result directory')
    parser.add_argument('--paired', required=False, action='store_true', default=False, help='Indicate whether the analysis is for paired prediction')
    parser.add_argument('--ct', required=True, nargs='+', default=[], help='Full cell type names, for paired this would be two cell types')
    parser.add_argument('--chrom', required=True, type=int, help='Chromosome number')
    parser.add_argument('--pred_len', required=False, type=int, default=200, help='Prediction length, default=200')
    parser.add_argument('--avg_stripe', required=False, action='store_true', help='Average V-stripe, default=False')

    parser.add_argument('--topdom_window', required=False, type=int, default=10, help='Window size for running TopDom, default=10')
    parser.add_argument('--topdom_cutoff', required=False, type=float, default=0, help='Cutoff for running TopDom, anything below will be set to cutoff, default=0')

    parser.add_argument('--kernel', required=False, type=str, default='diff', help='Kernel used when evaluating the similarity of two TopDom lists, default=diff')
    parser.add_argument('--similar_window', required=False, type=int, default=10, help='Window size for running sliding window Pearson Correlation, default=10')

    parser.add_argument('--bin_size', required=False, type=int, default=10000, help='Bin size when running ChromaFold, default=10kb=10000')
    parser.add_argument('--pattern', required=False, type=str, default=None, help='Filling behavior for TopDom score interpolation, default=None')

    parser.add_argument('--thresh_cutoff', required=False, type=float, default=0.6, help='Cutoff for selecting window with difference, default=0.6')
    parser.add_argument('--thresh_margin', required=False, type=int, default=1000, help='Margin of error used when extending window with difference, default=1000')

    parser.add_argument('--db_file', required=True, type=str, help='Database file directory')
    parser.add_argument('--gtf_file', required=False, type=str, default='gencode.vM10.basic.annotation.gtf',help='GTF file directory')
    parser.add_argument('--table', required=False, type=str, default='features', help='Table name for db query')
    parser.add_argument('--featuretype', required=False, type=str, default='gene', help='Feature types to select for db query')
    parser.add_argument('--filters', required=False, nargs='+', default=[], help='Attribute filters in database query, input each filter with \"key=value\" format')

    parser.add_argument('--num_plot', required=False, type=int, default=0, help='Number of plots to generate, from top significance, default=0')
    parser.add_argument('--out_dir', required=True, type=str, help='Output directory to store the db query result')
    parser.add_argument('--fig_dir', required=True, type=str, help='Figure directory to put the generated figures')

    args = parser.parse_args()

    paired = args.paired
    chrom = args.chrom

    print('Processing chr{}'.format(chrom))

    if paired:
        res = pairwise_difference(args)
    else:
        res = pipe_single(args)
    
    print('DONE')
    

