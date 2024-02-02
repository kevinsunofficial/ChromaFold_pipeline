import numpy as np
import pandas as pd
import scipy
import scipy.stats
from scipy.stats import ranksums, ks_2samp
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class Analyzer:

    def __init__(self, pred1, pred2, pred_diff):
        self.pred1 = pred1
        self.pred2 = pred2
        self.pred = pred_diff
        self.L = pred_diff.shape[0]
        self.tests = ['ranksums', 'ks_2samp']
        self.means = ['diff', 'abs_diff']

    def get_region(self, start, end):
        pass

    def score_region(self, region, region1, region2):
        if not region.size: 
            return np.full(len(self.tests), np.nan), np.full(len(self.means), np.nan)

        value = region[~np.isnan(region)]
        value1, value2 = region1[~np.isnan(region1)], region2[~np.isnan(region2)]
        p_values = [ranksums(value1, value2).pvalue, ks_2samp(value1, value2).pvalue]
        changes = [np.mean(value), np.mean(np.abs(value))]

        return p_values, changes


class GeneAnalyzer(Analyzer):

    def __init__(self, pred1, pred2, pred_diff, genes) -> None:
        super().__init__(pred1, pred2, pred_diff)
        self.genes = genes

    def get_region(self, pred, start, end):
        start_, end_ = start // 10**4, end // 10**4
        perimeter = int(np.log2(end_ - start_ + 1)) + 5
        region = []
        first = min(max(0, start_ - perimeter), self.L)
        last = max(0, min(end_ + 1 + perimeter, self.L))

        for i in range(first, last):
            left = pred[min(max(0, i - 200), first - 1):first - 1, i]
            right = pred[i, i+2:min(i + 201, self.L)]
            padded = np.concatenate((left, right), axis=None)
            padded = np.pad(padded, (0, 398 - len(padded)), mode='constant', constant_values=np.nan)
            region.append(padded)
        
        return np.array(region)

    def score_region(self, t, m):
        pvalues, changes = [], []
        for i in tqdm(range(self.genes.shape[0])):
            seqid, start, end, gene_name, gene_id = self.genes.iloc[i].tolist()
            region = self.get_region(self.pred, start, end)
            region1 = self.get_region(self.pred1, start, end)
            region2 = self.get_region(self.pred2, start, end)

            pvalue, change = super().score_region(region, region1, region2)
            pvalues.append(pvalue)
            changes.append(change)
        
        pvalues, changes = -np.log10(np.array(pvalues)), np.array(changes)
        genes_score = self.genes.copy()
        for i, test in enumerate(self.tests):
            genes_score[test] = pvalues[:, i]
        for i, mn in enumerate(self.means):
            genes_score[mn] = changes[:, i]

        genes_score = genes_score[
            ~genes_score.isin([np.nan, np.inf, -np.inf]).any(1)
        ].sort_values([m, t], ascending=False).reset_index(drop=True)

        return genes_score
    

class TADAnalyzer(Analyzer):

    def __init__(self, pred1, pred2, pred_diff, vertex):
        super().__init__(pred1, pred2, pred_diff)
        self.vertex = vertex

    def get_region(self, pred, start, end):
        region = pred[start:end + 1, start:end + 1]
        region[np.tril_indices(region.shape[0], k=1)] = np.nan

        return np.array(region)
    
    def score_region(self, t, m):
        pvalues, changes = [], []      
        for i in tqdm(range(self.vertex.shape[0])):
            seqid, start, end = self.vertex.iloc[i].tolist()
            region = self.get_region(self.pred, start, end)
            region1 = self.get_region(self.pred1, start, end)
            region2 = self.get_region(self.pred2, start, end)

            pvalue, change = super().score_region(region, region1, region2)
            pvalues.append(pvalue)
            changes.append(change)
        
        pvalues, changes = -np.log10(np.array(pvalues)), np.array(changes)
        tad_score = self.vertex.copy()
        for i, test in enumerate(self.tests):
            tad_score[test] = pvalues[:, i]
        for i, mn in enumerate(self.means):
            tad_score[mn] = changes[:, i]

        tad_score = tad_score[
            ~tad_score.isin([np.nan, np.inf, -np.inf]).any(1)
        ].sort_values(t, ascending=False).reset_index(drop=True)

        return tad_score
