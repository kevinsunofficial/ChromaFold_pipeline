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
from dataloader import DataLoader
from analyzer import GeneAnalyzer, TADAnalyzer
from utils import *

import warnings
warnings.filterwarnings('ignore')


def gene_analysis(args):
    loader = DataLoader(args.root_dir, args.pred_dir, args.chrom, args.annotation, args.genome)

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
        analyzer = GeneAnalyzer(pred_diff, genes)
        gene_scores = analyzer.score_region()
        
        query_dir = osp.join(args.out_dir, 'query')
        if not osp.exists(query_dir):
            os.makedirs(query_dir)
        gene_scores.to_csv(
            osp.join(query_dir, f'chr{args.chrom}_significant_genes.csv'),
            header=True, index=False)
    else:
        print('No valid query result')
    
    return gene_scores



if __name__ == '__main__':
    print('\nParsing arguments...')

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', required=True, type=str, help='ChromaFold multiome input directory')
    parser.add_argument('--pred_dir', required=True, type=str, help='ChromaFold prediction result directory')
    parser.add_argument('--ct', required=True, type=str, nargs='+', help='Full cell type names, can take 1, 2, or 3 cts')
    parser.add_argument('--chrom', required=True, type=str, help='Chromosome number')

    parser.add_argument('--annotation', required=True, type=str, help='Database file directory')
    parser.add_argument('--featuretype', required=False, type=str, default='gene', help='Feature types to select for db query')
    parser.add_argument('--filters', required=False, nargs='+', default=[], help='Attribute filters in database query, input each filter with \"key=value\" format')

    parser.add_argument('--out_dir', required=True, type=str, help='Output directory to store the pipeline result')

    args = parser.parse_args()

    res = gene_analysis(args)