import os
import os.path as osp
import numpy as np
import pandas as pd
import scipy
import scipy.stats
import gffutils
import argparse
import sqlite3
from dataloader import *
from metrics import *

import warnings
warnings.filterwarnings('ignore')


def pipe_single(args):
    pred_dir = args.pred_dir
    ct = args.ct
    chrom = args.chrom
    pred_len = args.pred_len
    avg_stripe = args.avg_stripe
    topdom_window_size = args.topdom_w
    bin_size = args.bin_size
    smooth = args.smooth

    pred_mat = load_pred(pred_dir, ct, chrom, pred_len=pred_len, avg_stripe=avg_stripe)
    raw_signal = topdom(pred_mat, window_size=topdom_window_size)
    signal = interpolate(raw_signal, bin_size=bin_size, smooth=smooth)

    return signal


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', required=True, type=str, help='ChromaFold prediction result directory')
    parser.add_argument('--ct', required=True, type=str, help='Full cell type name')
    parser.add_argument('--chrom', required=True, type=int, help='Chromosome number')
    parser.add_argument('--pred_len', required=False, type=int, default=200, help='Prediction length, default=200')
    parser.add_argument('--avg_stripe', required=False, action='store_true', help='Average V-stripe, default=False')

    parser.add_argument('--topdom_w', required=False, type=int, default=10, help='Window size for running TopDom, default=10')

    parser.add_argument('--bin_size', required=False, type=int, default=10000, help='Bin size when running ChromaFold, default=10kb=10000')
    parser.add_argument('--smooth', required=False, action='store_true', help='Use smooth interpolation for TopDop score, default=False')

    parser.add_argument('--gtf_file', required=False, type=str, default='gencode.vM10.basic.annotation.gtf',help='GTF file directory')
    parser.add_argument('--db_file', required=True, type=str, help='Database file directory')

    args = parser.parse_args()

    signal = pipe_single(args)
