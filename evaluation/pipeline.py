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


def sort_significance(args, res, valid, pred1, pred2, kernel, ranked=None):
    scores, directions = [], []
    for i in range(valid):
        start, _, _, _ = parse_res(res.iloc[i])
        if kernel == 'tad_diff':
            direction, score = match_tad_score(ranked, start)
        else:
            topdom_window_size = args.topdom_window
            topdom_cutoff = args.topdom_cutoff
            score = verification(
                pred1, pred2, start, 
                window_size=topdom_window_size, cutoff=topdom_cutoff)
            direction = 0
        scores.append(score)
        directions.append(direction)
    
    res['significance'] = scores
    res['directions'] = directions
    res = res.sort_values(by=['significance'], ignore_index=True, ascending=False)

    return res


def plot_gene(args, data, rank, start, locstart, locend, gene):
    fig_dir = args.fig_dir
    ct1, ct2 = args.ct[:2]

    ctcf, atac1, atac2, scatac_pre1, scatac_pre2, metacell1, metacell2, pred1, pred2 = data
    ctcf = ctcf[start*200: (start+700)*200]
    atac1, atac2 = atac1[start*200: (start+700)*200], atac2[start*200: (start+700)*200]
    scatac1 = process_scatac(scatac_pre1, metacell1, start)
    scatac2 = process_scatac(scatac_pre2, metacell2, start)

    savefig_dir = osp.join(fig_dir, f'chr{chrom}', f'{rank}_chr{chrom}_{gene}')
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

    print('Normalizing predictions...')
    pred1, pred2 = quantile_norm(pred1, pred2)
    data = [ctcf, atac1, atac2, scatac1, scatac2, metacell1, metacell2, pred1, pred2]

    print('Calculating TopDom insulation score...')
    signal1 = topdom(pred1, window_size=topdom_window_size, cutoff=topdom_cutoff)
    signal2 = topdom(pred2, window_size=topdom_window_size, cutoff=topdom_cutoff)

    print('Calculating Similarity...')
    raw_simscore = similarity(signal1, signal2, kernel=kernel, window_size=similar_window_size)
    simscore = interpolate(raw_simscore, bin_size=bin_size, pattern=pattern)

    print('Selecting significant regions...')
    regions = threshold(simscore, cutoff=thresh_cutoff, kernel=kernel, margin=thresh_margin)
    queries = generate_query(regions, chrom=chrom, table=table, featuretype=featuretype)

    print('Querying database...')
    db = load_database(db_file, gtf_file)
    res, numvalid = db_query(db, queries, filters=filters)

    if numvalid:
        print('Sorting query result by significance')
        res = sort_significance(args, res, numvalid, pred1, pred2, kernel)
        res.to_csv(osp.join(out_dir, f'chr{chrom}_significant_genes_{kernel}.csv'), header=True, index=False)
        print('Plotting result...')
        if numplot is not None and numplot > 0:
            for i in tqdm(range(numplot), desc='plotting result', position=0, leave=True):
                row = res.iloc[i]
                start, locstart, locend, gene = parse_res(row)
                plot_gene(args, data, i+1, start, locstart, locend, gene)

    return res


def pairwise_difference_tads(args):
    input_dir = args.input_dir
    pred_dir = args.pred_dir
    ct1, ct2 = args.ct[:2]
    chrom = args.chrom
    pred_len = args.pred_len
    avg_stripe = args.avg_stripe

    min_dim = args.min_dim
    max_dim = args.max_dim
    num_dim = args.num_dim
    close = args.close

    kernel = args.kernel
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

    print('Normalizing predictions...')
    pred1, pred2 = quantile_norm(pred1, pred2)
    data = [ctcf, atac1, atac2, scatac1, scatac2, metacell1, metacell2, pred1, pred2]

    print('Calculating TADs Similarity...')
    coords = get_tad_coords(pred1, pred2, min_dim=min_dim, max_dim=max_dim, num_dim=num_dim, close=close)
    ranked = rank_coords(pred1, pred2, coords)

    print('Querying database...')
    db = load_database(db_file, gtf_file)
    res, numvalid = db_query_tad(db, ranked, chrom=chrom, table=table, featuretype=featuretype, filters=filters)

    if numvalid:
        if not osp.exists(out_dir):
            os.makedirs(out_dir)
        res.to_csv(osp.join(out_dir, f'chr{chrom}_significant_genes_{kernel}.csv'), header=True, index=False)
        print('Plotting result...')
        if numplot is not None and numplot > 0:
            for i in tqdm(range(numplot), desc='plotting result', position=0, leave=True):
                row = res.iloc[i]
                start, locstart, locend, gene = parse_res(row)
                plot_gene(args, data, i+1, start, locstart, locend, gene)

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
    parser.add_argument('--topdom_cutoff', required=False, type=float, default=None, help='Cutoff for running TopDom, anything below will be set to cutoff, default=None')

    parser.add_argument('--min_dim', required=False, type=int, default=10, help='Minimum window size for running Region TopDom, default=10')
    parser.add_argument('--max_dim', required=False, type=int, default=100, help='Maximum window size for running Region TopDom, default=100')
    parser.add_argument('--num_dim', required=False, type=int, default=25, help='Number of window size for running Region TopDom, default=25')
    parser.add_argument('--close', required=False, type=int, default=5, help='Margin of error allowed for merging coordinates, default=5')

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
    kernel = args.kernel

    print('Processing chr{}'.format(chrom))

    if paired:
        if kernel == 'tad_diff':
            res = pairwise_difference_tads(args)
        else:
            res = pairwise_difference(args)
    else:
        res = pipe_single(args)
    
    print('DONE')
    

