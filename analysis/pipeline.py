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


def create_bedpe(args, ct, pred):
    bedpe_dir = osp.join(args.out_dir, 'bedpe')
    chrom = args.chrom
    bedpe_thresh = args.bedpe_thresh
    bedpe_margin = args.bedpe_margin

    if not osp.exists(bedpe_dir):
        os.makedirs(bedpe_dir)
    
    coords = parse_bedpe(pred, bedpe_thresh=bedpe_thresh, bedpe_margin=bedpe_margin)
    write_bedpe(coords, chrom, ct, bedpe_dir)

    return


def plot_gene(args, data, rank, start, locstart, locend, gene):
    fig_dir = osp.join(args.out_dir, 'figure')
    ct = args.ct[0]
    gtf_file = args.gtf_file
    bedpe_dir = osp.join(args.out_dir, 'bedpe')

    if not osp.exists(fig_dir):
        os.makedirs(fig_dir)

    ctcf, atac, scatac_pre, metacell, pred = data
    ctcf = ctcf[start*200: (start+700)*200]
    atac = atac[start*200: (start+700)*200]
    scatac = process_scatac(scatac_pre, metacell, start)

    savefig_dir = osp.join(fig_dir, f'chr{chrom}', f'{rank}_chr{chrom}_{gene}')
    if not osp.exists(savefig_dir):
        os.makedirs(savefig_dir)

    plot_ctcf(ctcf, chrom, start, gene, locstart, locend, savefig_dir)
    plot_atac(atac, ct, chrom, start, gene, locstart, locend, savefig_dir)
    plot_scatac(scatac, ct, chrom, start, gene, locstart, locend, savefig_dir)
    plot_pred(pred, ct, chrom, start, gene, locstart, locend, savefig_dir)
    plot_track(gtf_file, bedpe_dir, ct, chrom, start, gene, locstart, locend, savefig_dir)

    return


def plot_gene_paired(args, data, rank, start, locstart, locend, gene):
    fig_dir = osp.join(args.out_dir, 'figure')
    ct1, ct2 = args.ct[:2]
    gtf_file = args.gtf_file
    bedpe_dir = osp.join(args.out_dir, 'bedpe')

    if not osp.exists(fig_dir):
        os.makedirs(fig_dir)

    ctcf, atac1, atac2, scatac_pre1, scatac_pre2, metacell1, metacell2, pred1, pred2, pred_diff = data
    ctcf = ctcf[start*200: (start+700)*200]
    atac1, atac2 = atac1[start*200: (start+700)*200], atac2[start*200: (start+700)*200]
    scatac1 = process_scatac(scatac_pre1, metacell1, start)
    scatac2 = process_scatac(scatac_pre2, metacell2, start)

    savefig_dir = osp.join(fig_dir, f'chr{chrom}', f'{rank}_chr{chrom}_{gene}')
    if not osp.exists(savefig_dir):
        os.makedirs(savefig_dir)

    plot_ctcf(ctcf, chrom, start, gene, locstart, locend, savefig_dir)
    plot_atac_paired(atac1, atac2, ct1, ct2, chrom, start, gene, locstart, locend, savefig_dir)
    plot_scatac_paired(scatac1, scatac2, ct1, ct2, chrom, start, gene, locstart, locend, savefig_dir)
    plot_pred_paired(pred1, pred2, pred_diff, ct1, ct2, chrom, start, gene, locstart, locend, savefig_dir)
    plot_track_paired(gtf_file, bedpe_dir, ct1, ct2, chrom, start, gene, locstart, locend, savefig_dir)

    return


def single_significance(args):
    input_dir = args.input_dir
    pred_dir = args.pred_dir
    ct = args.ct[0]
    chrom = args.chrom

    min_dim = args.min_dim
    max_dim = args.max_dim
    num_dim = args.num_dim
    close = args.close

    db_file = args.db_file
    gtf_file = args.gtf_file
    table = args.table
    featuretype = args.featuretype
    filters = args.filters
    
    numplot = args.num_plot

    out_dir = args.out_dir
    if not osp.exists(out_dir):
        os.makedirs(out_dir)

    print('Loading multiome data...')
    ctcf, atac, scatac, metacell = load_multiome(input_dir, ct, chrom, start=None)

    print('Loading predictions...')
    pred = load_pred(pred_dir, ct, chrom)
    data = [ctcf, atac, scatac, metacell, pred]

    print('Calculating TADs...')
    coords = get_tad_coords(pred, min_dim=min_dim, max_dim=max_dim, num_dim=num_dim, close=close)
    ranked = rank_coords(pred, coords)

    print('Querying database...')
    db = load_database(db_file, gtf_file)
    res, numvalid = db_query_tad(db, ranked, chrom=chrom, table=table, featuretype=featuretype, filters=filters)

    print('Calculation and query is completed.\n\nGenerating results...')
    
    print('Writing BEDPE files...')
    create_bedpe(args, ct, pred)

    if numvalid:
        query_dir = osp.join(out_dir, 'query')
        if not osp.exists(query_dir):
            os.makedirs(query_dir)
        res.to_csv(osp.join(query_dir, f'chr{chrom}_significant_genes.csv'), header=True, index=False)
        
        if numplot is not None and numplot > 0:
            print('Plotting result...')
            for i in tqdm(range(numplot), desc='plotting result', position=0, leave=True):
                row = res.iloc[i]
                start, locstart, locend, gene = parse_res(row)
                plot_gene(args, data, i+1, start, locstart, locend, gene)
    else:
        print('No valid query result')

    return res


def pairwise_significance(args):
    input_dir = args.input_dir
    pred_dir = args.pred_dir
    ct1, ct2 = args.ct[:2]
    ctc = None if len(args.ct) == 2 else args.ct[-1] # control
    chrom = args.chrom

    min_dim = args.min_dim
    max_dim = args.max_dim
    num_dim = args.num_dim
    close = args.close

    db_file = args.db_file
    gtf_file = args.gtf_file
    table = args.table
    featuretype = args.featuretype
    filters = args.filters

    numplot = args.num_plot

    out_dir = args.out_dir
    if not osp.exists(out_dir):
        os.makedirs(out_dir)

    print('Loading multiome data...')
    ctcf, atac1, scatac1, metacell1 = load_multiome(input_dir, ct1, chrom, start=None)
    _, atac2, scatac2, metacell2 = load_multiome(input_dir, ct2, chrom, start=None)

    print('Loading predictions...')
    pred1 = load_pred(pred_dir, ct1, chrom)
    pred2 = load_pred(pred_dir, ct2, chrom)
    predc = None if ctc is None else load_pred(pred_dir, ctc, chrom)

    print('Normalizing predictions...')
    allpred = np.array([pred1, pred2]) if predc is None else np.array([pred1, pred2, predc])
    preds_qn = quantile_normalize(allpred)
    pred1, pred2 = preds_qn[0], preds_qn[1]
    pred_diff = pred1 - pred2
    data = [ctcf, atac1, atac2, scatac1, scatac2, metacell1, metacell2, pred1, pred2, pred_diff]

    print('Calculating TADs Similarity...')
    coords = get_tad_coords(pred_diff, min_dim=min_dim, max_dim=max_dim, num_dim=num_dim, close=close)
    ranked = rank_coords(pred_diff, coords)

    print('Querying database...')
    db = load_database(db_file, gtf_file)
    res, numvalid = db_query_tad(db, ranked, chrom=chrom, table=table, featuretype=featuretype, filters=filters)

    print('Calculation and query is completed.\n\nGenerating results...')
    
    print('Writing BEDPE files...')
    create_bedpe(args, ct1, pred1)
    create_bedpe(args, ct2, pred2)

    if numvalid:
        query_dir = osp.join(out_dir, 'query')
        if not osp.exists(query_dir):
            os.makedirs(query_dir)
        res.to_csv(osp.join(query_dir, f'chr{chrom}_significant_genes.csv'), header=True, index=False)
        
        if numplot is not None and numplot > 0:
            print('Plotting result...')
            for i in tqdm(range(numplot), desc='plotting result', position=0, leave=True):
                row = res.iloc[i]
                start, locstart, locend, gene = parse_res(row)
                plot_gene_paired(args, data, i+1, start, locstart, locend, gene)
    else:
        print('No valid query result')

    return res


def debugger(args):
    pass


if __name__=='__main__':

    print('\nParsing arguments...')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, type=str, help='ChromaFold multiome input directory')
    parser.add_argument('--pred_dir', required=True, type=str, help='ChromaFold prediction result directory')
    parser.add_argument('--ct', required=True, type=str, nargs='+', help='Full cell type names, can take 1, 2, or 3 cts')
    parser.add_argument('--chrom', required=True, type=str, help='Chromosome number')

    parser.add_argument('--min_dim', required=False, type=int, default=10, help='Minimum window size for running Region TopDom, default=10')
    parser.add_argument('--max_dim', required=False, type=int, default=100, help='Maximum window size for running Region TopDom, default=100')
    parser.add_argument('--num_dim', required=False, type=int, default=25, help='Number of window size for running Region TopDom, default=25')
    parser.add_argument('--close', required=False, type=int, default=5, help='Margin of error allowed for merging coordinates, default=5')

    parser.add_argument('--db_file', required=True, type=str, help='Database file directory')
    parser.add_argument('--gtf_file', required=True, type=str, default='gencode.vM10.basic.annotation.gtf',help='GTF file directory')
    parser.add_argument('--table', required=False, type=str, default='features', help='Table name for db query')
    parser.add_argument('--featuretype', required=False, type=str, default='gene', help='Feature types to select for db query')
    parser.add_argument('--filters', required=False, nargs='+', default=[], help='Attribute filters in database query, input each filter with \"key=value\" format')

    parser.add_argument('--num_plot', required=False, type=int, default=0, help='Number of plots to generate, from top significance, default=0')
    parser.add_argument('--bedpe_thresh', required=False, type=float, default=99., help='Percentile or absolute threshold for bedpe generation, default=99')
    parser.add_argument('--bedpe_margin', required=False, type=int, default=None, help='Margin of error used when extending bedpe, default=None')
    parser.add_argument('--out_dir', required=True, type=str, help='Output directory to store the pipeline result')

    parser.add_argument('--debug', required=False, action='store_true', default=False, help='Toggling debugging mode for certain functionality')

    args = parser.parse_args()

    chrom = args.chrom
    ct = args.ct

    print('Processing chr{}'.format(chrom))

    if args.debug:
        debugger(args)
        exit()

    if len(ct) > 1:
        res = pairwise_significance(args)
    else:
        res = single_significance(args)
    
    print('Done')
