import os
import os.path as osp
import numpy as np
import pandas as pd
import scipy
import scipy.stats
from scipy.stats import kstest, ks_2samp
from scipy.signal import find_peaks
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class Analyzer:

    def __init__(self, pred1, pred2, pred_diff):
        self.pred1 = pred1
        self.pred2 = pred2
        self.pred = pred_diff
        self.L = pred_diff.shape[0]

    def get_region(self, start, end):
        pass

    def score_region(self, region, region1, region2):
        if not region.size:
            return np.nan, np.nan

        if np.nanmean(region1) < .33 and np.nanmean(region2) < .33:
            return np.nan, np.nan
        
        lower, upper = -1, 1
        value = region[~np.isnan(region)]
        value_z = (value - np.mean(value)) / np.std(value)
        value_sig = value_z[(value_z < lower) | (value_z > upper)]
        normal = np.random.normal(0, 1, region.size)
        normal_sig = normal[(normal < lower) | (normal > upper)]

        pvals = []
        np.random.seed(100)
        for _ in range(100):
            value_choice = np.random.choice(value_sig, size=1000, replace=True)
            normal_choice = np.random.choice(normal_sig, size=1000, replace=True)
            stat, pval = ks_2samp(value_choice, normal_choice)
            pvals.append(pval)
        
        neglog10pval, change = -np.log10(np.mean(pvals)), np.mean(value_sig)
        
        return neglog10pval, change


class GeneAnalyzer(Analyzer):

    def __init__(self, pred1, pred2, pred_diff, genes) -> None:
        super().__init__(pred1, pred2, pred_diff)
        self.genes = genes

    def get_region(self, pred, start, end):
        start_, end_ = start // 10**4, end // 10**4
        perimeter = int(np.log(end - start)) + 5
        region = []

        first = min(max(0, start_ - perimeter), self.L)
        last = max(0, min(end_ + 1 + perimeter, self.L))
        
        for i in range(first, last):
            left_margin = (max(0, 200 - i), min(i - first, 200))
            left = pred[max(0, i - 200):first - 1, i]
            left_pad = np.pad(left, left_margin, mode='constant', constant_values=np.nan)
            right_margin = (0, max(0, i + 201 - self.L))
            right = pred[i, i + 2:min(i + 201, self.L)]
            right_pad = np.pad(right, right_margin, mode='constant', constant_values=np.nan)
            region.append(np.array([left_pad, right_pad]).flatten())
        
        return np.array(region)
    
    def score_region(self):
        neglog10pval, changes = [], []        
        for i in tqdm(range(self.genes.shape[0])):
            seqid, start, end, gene_name, gene_id = self.genes.iloc[i]
            region = self.get_region(self.pred, start, end)
            region1 = self.get_region(self.pred1, start, end)
            region2 = self.get_region(self.pred2, start, end)

            pval, change = super().score_region(region, region1, region2)
            neglog10pval.append(pval)
            changes.append(change)
        
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

    def __init__(self, pred1, pred2, pred_diff, vertex):
        super().__init__(pred1, pred2, pred_diff)
        self.vertex = vertex

    def get_region(self, pred, start, end):
        region = pred[start:end + 1, start:end + 1]
        region[np.tril_indices(region.shape[0], k=1)] = np.nan

        return np.array(region)
    
    def score_region(self):
        neglog10pval, changes = [], []        
        for i in tqdm(range(self.vertex.shape[0])):
            start, end = self.vertex.iloc[i]
            region = self.get_region(self.pred, start, end)
            region1 = self.get_region(self.pred1, start, end)
            region2 = self.get_region(self.pred2, start, end)

            pval, change = super().score_region(region, region1, region2)
            neglog10pval.append(pval)
            changes.append(change)
        
        vertex_score = self.vertex.copy()
        vertex_score['neglog10pval'] = neglog10pval
        vertex_score['changes'] = changes
        vertex_score = vertex_score[
            vertex_score.neglog10pval.notna() & vertex_score.changes.notna()
        ].sort_values(
            ['neglog10pval', 'changes'], ascending=False
        ).reset_index(drop=True)

        return vertex_score
