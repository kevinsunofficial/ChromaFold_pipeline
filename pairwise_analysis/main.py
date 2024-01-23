import os
import os.path as osp
import numpy as np
import pandas as pd
import scipy
import scipy.stats
import argparse
from dataloader import DataLoader
from analyzer import GeneAnalyzer, TADAnalyzer
from plotgenerator import PairGenePlotGenerator, PairTADPlotGenerator
from utils import *

import warnings
warnings.filterwarnings('ignore')


def gene_analysis(args):
    print('Loading data...')
    loader = DataLoader(args.root_dir, args.pred_dir, args.chrom, args.annotation)

    ct1, ct2 = args.ct[:2]
    ctc = None if len(args.ct) == 2 else args.ct[-1]

    pred1, pred2 = loader.load_pred(ct1), loader.load_pred(ct2)
    predc = None if ctc is None else loader.load_pred(ctc)

    allpred = np.array([pred1, pred2]) if predc is None else np.array([pred1, pred2, predc])
    preds_qn = quantile_normalize(allpred)
    pred1, pred2 = preds_qn[0], preds_qn[1]
    pred_diff = pred1 - pred2

    db = loader.load_database()
    numvalid, genes = db_query(db, args.chrom)

    if numvalid:
        print('Analyzing...')
        analyzer = GeneAnalyzer(pred1, pred2, pred_diff, genes)
        gene_score = analyzer.score_region(args.test, args.mean) # it is very unlikely that this is empty
        
        query_dir = osp.join(args.out_dir, 'query')
        if not osp.exists(query_dir):
            os.makedirs(query_dir)
        gene_score.to_csv(
            osp.join(query_dir, f'chr{args.chrom}_significant_genes.csv'),
            header=True, index=False)

        fig_dir = osp.join(args.out_dir, 'figure', args.mode, f'chr{args.chrom}')
        if not osp.exists(fig_dir):
            os.makedirs(fig_dir)
        bedpe_dir = osp.join(args.out_dir, 'bedpe', f'chr{args.chrom}')
        if not osp.exists(bedpe_dir):
            os.makedirs(bedpe_dir)
        gtf_file = f'{args.annotation}.gtf'

        create_bedpe(pred1, args.chrom, ct1, bedpe_dir)
        create_bedpe(pred2, args.chrom, ct2, bedpe_dir)

        print('Plotting results...')
        plotgenerator = PairGenePlotGenerator(
            gene_score,
            fig_dir, bedpe_dir, gtf_file, args.chrom, 
            ct1, ct2, pred1, pred2, pred_diff,
            ctcf=loader.load_ctcf(), atac=loader.load_atac(ct1), scatac=None,
            atac2=loader.load_atac(ct2), scatac2=None
        )
        plotgenerator.plot_genes(args.numplot)
        for gene in args.extraplot:
            plotgenerator.plot_gene(gene=gene)
        plotgenerator.volcano_plot(args.test, args.mean)
        
    else:
        print('No valid query result')
    
    return gene_score


def tad_analysis(args):
    print('Loading data...')
    loader = DataLoader(args.root_dir, args.pred_dir, args.chrom, args.annotation)

    ct1, ct2 = args.ct[:2]
    ctc = None if len(args.ct) == 2 else args.ct[-1]

    pred1, pred2 = loader.load_pred(ct1), loader.load_pred(ct2)
    predc = None if ctc is None else loader.load_pred(ctc)

    allpred = np.array([pred1, pred2]) if predc is None else np.array([pred1, pred2, predc])
    preds_qn = quantile_normalize(allpred)
    pred1, pred2 = preds_qn[0], preds_qn[1]
    pred_diff = pred1 - pred2

    vertex = get_tad_vertex(pred_diff)  # it is very unlikely that this is empty

    print('Analyzing...')
    analyzer = TADAnalyzer(pred1, pred2, pred_diff, vertex)
    tad_score = analyzer.score_region(args.test, args.mean)  # it is very unlikely that this is empty

    query_dir = osp.join(args.out_dir, 'query')
    if not osp.exists(query_dir):
        os.makedirs(query_dir)
    tad_score.to_csv(
        osp.join(query_dir, f'chr{args.chrom}_significant_TADs.csv'),
        header=True, index=False)

    fig_dir = osp.join(args.out_dir, 'figure', args.mode, f'chr{args.chrom}')
    if not osp.exists(fig_dir):
        os.makedirs(fig_dir)
    bedpe_dir = osp.join(args.out_dir, 'bedpe', f'chr{args.chrom}')
    if not osp.exists(bedpe_dir):
        os.makedirs(bedpe_dir)
    gtf_file = f'{args.annotation}.gtf'

    create_bedpe(pred1, args.chrom, ct1, bedpe_dir)
    create_bedpe(pred2, args.chrom, ct2, bedpe_dir)

    print('Plotting results...')
    plotgenerator = PairTADPlotGenerator(
        tad_score,
        fig_dir, bedpe_dir, gtf_file, args.chrom, 
        ct1, ct2, pred1, pred2, pred_diff,
        ctcf=loader.load_ctcf(), atac=loader.load_atac(ct1), scatac=None,
        atac2=loader.load_atac(ct2), scatac2=None
    )
    plotgenerator.plot_tads(args.numplot)
    plotgenerator.volcano_plot(args.test, args.mean)

    return tad_score



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', required=True, type=str, help='ChromaFold multiome input directory')
    parser.add_argument('--pred_dir', required=True, type=str, help='ChromaFold prediction result directory')
    parser.add_argument('--out_dir', required=True, type=str, help='Output directory to store the pipeline result')
    parser.add_argument('--annotation', required=True, type=str, help='Annotation file name')
    parser.add_argument('--ct', required=True, type=str, nargs='+', help='Full cell type names, can take 1, 2, or 3 cts')
    parser.add_argument('--chrom', required=True, type=str, help='Chromosome number')
    parser.add_argument('--mode', required=True, type=str, help='Mode for the analysis, can choose from \"gene\" or \"tad\"')
    
    parser.add_argument('--featuretype', required=False, type=str, default='gene', help='Feature types to select for db query')
    parser.add_argument('--filters', required=False, nargs='+', default=[], help='Attribute filters in database query, input each filter with \"key=value\" format')
    parser.add_argument('--test', required=False, default='ranksums', help='Statistical test to use for scoring regions')
    parser.add_argument('--mean', required=False, default='abs_diff', help='Type of difference to average for scoring regions')

    parser.add_argument('--numplot', required=False, type=int, default=10, help='Number of plots to generate from top significance')
    parser.add_argument('--extraplot', required=False, nargs='+', default=[], help='Genes to plot if they are not top significance')

    args = parser.parse_args()

    print(f'\nANALYSIS BEGIN - chr{args.chrom}')
    if args.mode == 'gene':
        res = gene_analysis(args)
    elif args.mode == 'tad':
        res = tad_analysis(args)
    else:
        raise ValueError
    
    print('ANALYSIS COMPLETE')
    