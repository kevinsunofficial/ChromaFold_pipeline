import os
import os.path as osp
import numpy as np
import pandas as pd
import scipy
import torch
from scipy.sparse import csr_matrix
import pickle
import gffutils
from tqdm import tqdm
import warnings
import sqlite3
import json

warnings.filterwarnings('ignore')


class DataLoader:

    def __init__(self, root_dir, pred_dir, chrom, annotation, genome='mm10'):
        self.root_dir = root_dir
        self.pred_dir = pred_dir
        self.chrom = chrom
        self.annotation = annotation
        self.genome = genome
    
    def load_ctcf(self, start=None):
        ctcf_path = osp.join(self.root_dir, 'dna', f'{self.genome}_ctcf_motif_score.p')
        ctcf = pickle.load(open(ctcf_path, 'rb'))[f'chr{self.chrom}'].toarray()[0]
        if start is not None:
            ctcf = ctcf[start*200: (start+700)*200]
        
        return ctcf
    
    def load_atac(self, ct, start=None):
        atac_path = osp.join(self.root_dir, 'atac', f'{ct}_tile_pbulk_50bp_dict.p')
        atac = pickle.load(open(atac_path, 'rb'))[f'chr{self.chrom}'].flatten()
        if start is not None:
            atac = atac[start*200, (start+700)*200]

        return atac
    
    @staticmethod
    def process_scatac(scatac_pre, metacell, start):
        tmp = torch.tensor((metacell * scatac_pre)[:, start*20:(start+700)*20].toarray()).T

        size, eps = tmp.shape[1], 1e-8
        one, zero = torch.tensor(1.0), torch.tensor(0.0)
        lrg = torch.where(tmp>0, one, zero)
        eql = torch.where(tmp==0, one, zero)
        num, denom = lrg @ lrg.T, size - eql @ eql.T
        scatac = torch.div(num, torch.max(denom, eps * torch.ones_like(denom)))
        scatac[scatac != scatac] = 0

        scatac = scatac.reshape(
            scatac.shape[0]//20, 20, -1
        ).mean(axis=1).reshape(
            -1, scatac.shape[1]//20, 20
        ).mean(axis=2)

        return scatac
    
    def load_scatac(self, ct, start=None):
        scatac_path = osp.join(self.root_dir, 'atac', f'{ct}_tile_500bp_dict.p')
        metacell_path = osp.join(self.root_dir, 'atac', f'{ct}_metacell_mask.csv')
        scatac_pre = pickle.load(open(scatac_path, 'rb'))[f'chr{self.chrom}']
        metacell = csr_matrix(pd.read_csv(metacell_path, index_col=0).values)

        if start is not None:
            scatac = self.process_scatac(scatac_pre, metacell, start)
            return scatac, metacell
        else:
            return scatac_pre, metacell
        
    @staticmethod
    def set_diagonal(mat, value=0):
        assert mat.shape[0] == mat.shape[1], f'Matrix is not square {mat.shape}'
        l = mat.shape[0]
        idx = np.arange(l)
        mat[idx[:-1], idx[1:]], mat[idx[1:], idx[:-1]] = value, value

        return mat
    
    def load_pred(self, ct, pred_len=200, avg_stripe=True):
        pred_path = osp.join(self.pred_dir, ct, f'prediction_{ct}_chr{self.chrom}.npz')
        temp = np.load(pred_path)['arr_0']
        chrom_len = temp.shape[0]

        prep = np.insert(temp, pred_len, 0, axis=1)
        mat = np.array([
            np.insert(np.zeros(chrom_len+pred_len+1), i, prep[i]) for i in range(chrom_len)
        ])
        summed = np.vstack((
            np.zeros((pred_len, mat.shape[1])), mat
        )).T[:chrom_len+pred_len, :chrom_len+pred_len]
        if avg_stripe:
            summed = (summed + np.vstack((
                np.zeros((pred_len, mat.shape[1])), mat
            ))[:chrom_len+pred_len, :chrom_len+pred_len])/2
        
        pred = self.set_diagonal(summed[pred_len:-pred_len, pred_len:-pred_len])

        return pred
    
    def load_database(self):
        db_path = f'{self.annotation}.db'
        if osp.isfile(db_path):
            db = gffutils.FeatureDB(db_path)
        else:
            print('Creating DB from raw. This might take a while...')
            gtf_path = f'{self.annotation}.gtf'
            db = gffutils.create_db(gtf_path, db_path)
        
        return db
