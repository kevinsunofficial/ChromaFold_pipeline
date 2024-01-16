import os
import os.path as osp
import numpy as np
import pandas as pd
import scipy
import scipy.stats
from scipy.stats import kstest
from scipy.signal import find_peaks
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class Analyzer:

    def __init__(self, pred_diff):
        self.pred = pred_diff
        self.L = pred_diff.shape[0]

    def get_region(self, start, end):
        pass

    def score_region(self):
        pass


class GeneAnalyzer(Analyzer):

    def __init__(self, pred_diff, genes) -> None:
        super(Analyzer).__init__(pred_diff)
        self.genes = genes

    def get_region(self, start, end):
        start_, end_ = start // 10**4, end // 10**4
        perimeter = int(np.log(end - start)) + 5
        region = []

        first = min(max(0, start_ - perimeter), self.L)
        last = max(0, min(end_ + 1 + perimeter, self.L))
        
        for i in range(first, last):
            left_margin = (max(0, 200 - i), min(i - first, 200))
            left = self.pred[max(0, i - 200):first - 1, i]
            left_pad = np.pad(left, left_margin, mode='constant', constant_values=np.nan)
            right_margin = (0, max(0, i + 201 - self.L))
            right = self.pred[i, i + 2:min(i + 201, self.L)]
            right_pad = np.pad(right, right_margin, mode='constant', constant_values=np.nan)
            region.append(np.array([left_pad, right_pad]).flatten())
        
        return np.array(region)
    
    def score_region(self):
        neglog10pval, changes = [], []        
        for i in range(self.genes.shape[0]):
            seqid, start, end, gene_name, gene_id = self.genes.iloc[i]
            region = self.get_region(start, end)

            if not region.size:
                continue

            value = region[~np.isnan(region)]
            value_norm = (value - np.mean(value)) / np.std(value)
            pvals = []
            np.random.seed(100)
            for _ in range(100):
                value_choice = np.random.choice(value_norm, size=1000, replace=True)
                statistics, p_value = kstest(value_choice, 'norm')
                pvals.append(p_value)

            neglog10pval.append(-np.log10(np.mean(pvals)))
            
            lower, upper = np.percentile(value, [5, 95])
            value_sig = value[(value < lower) | (value > upper)]
            changes.append(np.mean(value_sig))
        
        genes_score = self.genes.copy()
        genes_score['neglog10pval'] = neglog10pval
        genes_score['changes'] = changes
        genes_score = genes_score[
            genes_score.neglog10pval.notna() & genes_score.changes.notna()
        ].sort_values(
            ['neglog10pval', 'changes'], ascending=False
        ).reset_index(drop=True)

        return genes_score
    

class TADAnalyzer(Analyzer):

    def __init__(self, pred_diff, coords):
        super(Analyzer).__init__(pred_diff)
        self.coords = coords

    def get_region(self, start, end):
        region = self.pred[start:end + 1, start:end + 1]

        return np.array(region)
    
    def score_region(self):
        neglog10pval, changes = [], []        
        for i in range(self.coords.shape[0]):
            start, end, width = self.coords.iloc[i]
            region = self.get_region(start, end)

            if not region.size:
                continue

            value = region[~np.isnan(region)]
            value_norm = (value - np.mean(value)) / np.std(value)
            pvals = []
            np.random.seed(100)
            for _ in range(100):
                value_choice = np.random.choice(value_norm, size=1000, replace=True)
                statistics, p_value = kstest(value_choice, 'norm')
                pvals.append(p_value)

            neglog10pval.append(-np.log10(np.mean(pvals)))
            
            lower, upper = np.percentile(value, [5, 95])
            value_sig = value[(value < lower) | (value > upper)]
            changes.append(np.mean(value_sig))
        
        coords_core = self.coords.copy()
        coords_core['neglog10pval'] = neglog10pval
        coords_core['changes'] = changes
        coords_core = coords_core[
            coords_core.neglog10pval.notna() & coords_core.changes.notna()
        ].sort_values(
            ['neglog10pval', 'changes'], ascending=False
        ).reset_index(drop=True)

        return coords_core
