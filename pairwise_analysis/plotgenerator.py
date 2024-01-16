import os
import os.path as osp
import numpy as np
import pandas as pd
import scipy
from scipy.sparse import csr_matrix
import torch
from sklearn import metrics
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
import coolbox
from coolbox.api import *
import warnings

warnings.filterwarnings('ignore')


def rotate_coord(n):
    tmp = np.array(list(itertools.product(range(n,-1,-1),range(0,n+1,1))))
    tmp[:, [0, 1]] = tmp[:, [1, 0]]
    A = tmp.dot(np.array([[1, 0.5], [-1, 0.5]]))
    
    return A


class SinglePlotGenerator():

    def __init__(self, fig_dir, bedpe_dir, gtf_file, chrom, 
                 ct, pred, ctcf=None, atac=None, scatac=None):
        self.fig_dir = fig_dir
        self.bedpe_dir = bedpe_dir
        self.gtf_file = gtf_file
        self.chrom = chrom
        self.ct = ct
        self.pred = pred
        self.ctcf = ctcf
        self.atac = atac
        self.scatac = scatac

    def plot_ctcf(self, start, gene, locstart, locend):
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.plot(self.ctcf)
        plt.xlim(300*200, 500*200)
        plt.ylim(2.5, 3.5)
        tickloc = ax.get_xticks()
        ticklabel = np.linspace(start+300, start+500, num=len(tickloc), dtype=int).tolist()
        plt.xticks(tickloc, ticklabel)
        plt.title('chr{} - {}'.format(self.chrom, gene), fontsize=15)
        plt.xlabel('chr{} (10kb)'.format(self.chrom))
        plt.ylabel('CTCF motif score')
        plt.axvspan(locstart*200, locend*200, alpha=0.3, color='red')
        plt.savefig(osp.join(self.fig_dir, 'chr{}_{}_ctcf.png'.format(self.chrom, gene)))
        plt.clf()

    def plot_atac(self, start, gene, locstart, locend):
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.plot(self.atac)
        plt.xlim(300*200, 500*200)
        tickloc = ax.get_xticks()
        ticklabel = np.linspace(start+300, start+500, num=len(tickloc), dtype=int).tolist()
        plt.xticks(tickloc, ticklabel)
        plt.title('chr{} - {} ({})'.format(self.chrom, gene, self.ct), fontsize=15)
        plt.xlabel('chr{} (10kb)'.format(self.chrom))
        plt.ylabel('ATAC-seq signal')
        plt.axvspan(locstart*200, locend*200, alpha=0.3, color='red')
        plt.savefig(osp.join(self.fig_dir, 'chr{}_{}_atac.png'.format(self.chrom, gene)))
        plt.clf()

    def plot_scatac(self, start, gene, locstart, locend):
        fig, ax = plt.subplots(figsize=(8, 3))
        vmax, vmin = 0.5, 0
        n = self.scatac.shape[0]
        A = rotate_coord(n)
        img = ax.pcolormesh(A[:, 1].reshape(n+1, n+1), A[:, 0].reshape(n+1, n+1), 
                            np.flipud(self.scatac), cmap='RdBu_r', vmax=vmax, vmin=vmin)
        fig.colorbar(img)
        ax.set_xlim(300, 500)
        ax.set_ylim(0, 200)

        x = np.arange(300, 500)
        y1, y2 = np.absolute(2*x-2*locstart), np.absolute(2*x-2*locend)
        ax.plot(x, y1, color='magenta', alpha=0.3)
        ax.plot(x, y2, color='magenta', alpha=0.3)
        
        tickloc = ax.get_xticks()
        ticklabel = np.linspace(start+300, start+500, num=len(tickloc),dtype=int).tolist()
        plt.xticks(tickloc, ticklabel)
        plt.title('chr{} - {} ({})'.format(self.chrom, gene, self.ct), fontsize=15)
        plt.ylabel('Co-accessibility')
        plt.savefig(osp.join(self.fig_dir, 'chr{}_{}_scatac.png'.format(self.chrom, gene)))
        plt.clf()

    def plot_pred(self, start, gene, locstart, locend):
        fig, ax = plt.subplots(figsize=(8,3))
        vmax, vmin = 4, -1
        pred = np.triu(self.pred[start:start+700, start:start+700])
        n = pred.shape[0]
        A = rotate_coord(n)
        img = ax.pcolormesh(A[:, 1].reshape(n+1, n+1), A[:, 0].reshape(n+1, n+1), 
                            np.flipud(pred), cmap='RdBu_r', vmax=vmax, vmin=vmin)
        fig.colorbar(img)
        ax.set_xlim(300, 500)
        ax.set_ylim(0, 200)

        x = np.arange(300, 500)
        y1, y2 = np.absolute(2*x-2*locstart), np.absolute(2*x-2*locend)
        ax.plot(x, y1, color='magenta', alpha=0.3)
        ax.plot(x, y2, color='magenta', alpha=0.3)
        
        tickloc = ax.get_xticks()
        ticklabel = np.linspace(start+300, start+500, num=len(tickloc),dtype=int).tolist()
        plt.xticks(tickloc, ticklabel)
        plt.title('chr{} - {} ({})'.format(self.chrom, gene, self.ct), fontsize=12)
        plt.ylabel('Prediction Z-score')
        plt.savefig(osp.join(self.fig_dir, 'chr{}_{}_pred.png'.format(self.chrom, gene)))
        plt.clf()

    def plot_track(self, start, gene, locstart, locend):
        swstr = '0.05+score**2/10'
        hlstart, hlend = int((locstart + start) * 1e4), int((locend + start) * 1e4)
        hlregion = [f'chr{self.chrom}:{hlstart}-{hlend}']
        gtfs = GTF(self.gtf_file, length_ratio_thresh=0.005, fontsize=32, height=5) + Title('GTF Annotation')
        arc = Arcs(
            osp.join(self.bedpe_dir, f'chr{self.chrom}.bedpe'),
            linewidth=None, score_to_width=swstr
        ) + Inverted() + TrackHeight(8) + Title(f'chr{self.chrom}:{gene}')
        hl = HighLights(hlregion, color='red', alpha=0.1)

        frame = XAxis() + gtfs + hl + Spacer(0.5) + arc + hl
        test_range = f'chr{self.chrom}:{(start+300)*200*50}-{(start+500)*200*50}'
        frame.plot(test_range).savefig(osp.join(self.fig_dir, f'{self.ct}_chr{self.chrom}_{gene}_bedpe.png'))
        plt.clf()


class PairPlotGenerator(SinglePlotGenerator):

    def __init__(self, fig_dir, bedpe_dir, gtf_file, chrom, 
                 ct, ct2, pred, pred2, pred_diff,
                 ctcf=None, atac=None, scatac=None,
                 atac2=None, scatac2=None):
        super(SinglePlotGenerator).__init__(
            fig_dir, bedpe_dir, gtf_file, chrom, 
            ct, pred, ctcf, atac, scatac)
        self.ct2 = ct2
        self.pred2 = pred2
        self.pred_diff = pred_diff
        self.atac2 = atac2
        self.scatac2 = scatac2

    def plot_ctcf(self, start, gene, locstart, locend):
        return super().plot_ctcf(start, gene, locstart, locend)
    
    def plot_atac(self, start, gene, locstart, locend):
        plt.rcParams['figure.figsize'] = 8, 5
        plt.rcParams['figure.autolayout'] = False
        
        fig, axs = plt.subplots(3, 1, sharex=True, sharey=True)
        fig.suptitle('chr{} - {}'.format(self.chrom, gene), fontsize=15)
        fig.tight_layout(rect=[0.01, 0.03, 1, 4.5/5], h_pad=2)
        
        atacs = [self.atac, self.atac2, self.atac-self.atac2]
        cts = [self.ct, self.ct2, 'Difference']
        for i in range(3):
            ax, atac, ct = axs[i], atacs[i], cts[i]
            ax.plot(atac)
            ax.title.set_text(ct)
            ax.set_xlim(300*200, 500*200)
            ax.set_xticklabels(np.linspace(start+300, start+500, num=len(ax.get_xticks()), dtype=int))
            ax.axvspan(locstart*200, locend*200, alpha=0.3, color='red')
        
        axs[1].set_ylabel('ATAC-seq signal')
        plt.xlabel('chr{} (10kb)'.format(self.chrom))
        plt.savefig(osp.join(self.fig_dir, 'chr{}_{}_atac.png'.format(self.chrom, gene)))
        plt.clf()

    def plot_scatac(self, start, gene, locstart, locend):
        plt.rcParams['figure.figsize'] = 8, 6.5
        plt.rcParams['figure.autolayout'] = False

        fig, axs = plt.subplots(2, 1, sharex=True)
        fig.suptitle('chr{} - {}'.format(self.chrom, gene), fontsize=15)
        fig.tight_layout(rect=[0.01, 0.03, 1, 6/6.5], h_pad=2)
        
        vmax, vmin = 0.5, 0
        scatacs = [self.scatac, self.scatac2]
        cts = [self.ct, self.ct2]
        
        for i in range(2):
            ax, scatac, ct = axs[i], scatacs[i], cts[i]
            n = scatac.shape[0]
            A = rotate_coord(n)
            img = ax.pcolormesh(A[:, 1].reshape(n+1, n+1), A[:, 0].reshape(n+1, n+1),
                                np.flipud(scatac), cmap='RdBu_r', vmax=vmax, vmin=vmin)
            ax.title.set_text(ct)
            ax.set_xlim(300, 500)
            ax.set_ylim(0, 200)
            ax.set_xticklabels(np.linspace(start+300, start+500, num=len(ax.get_xticks()), dtype=int))
            
            x = np.arange(300, 500)
            y1, y2 = np.absolute(2*x-2*locstart), np.absolute(2*x-2*locend)
            ax.plot(x, y1, color='magenta', alpha=0.3)
            ax.plot(x, y2, color='magenta', alpha=0.3)
        
        fig.colorbar(img, ax=axs, shrink=1/2)
        plt.xlabel('chr{} (10kb)'.format(self.chrom))
        plt.savefig(osp.join(self.fig_dir, 'chr{}_{}_scatac.png'.format(self.chrom, gene)))
        plt.clf()

    def plot_pred(self, start, gene, locstart, locend):
        plt.rcParams['figure.figsize'] = 8, 9.5
        plt.rcParams['figure.autolayout'] = False

        fig, axs = plt.subplots(3, 1, sharex=True)
        fig.suptitle('chr{} - {}'.format(self.chrom, gene), fontsize=15)
        fig.tight_layout(rect=[0.01, 0.03, 1, 9/9.5], h_pad=2)
        
        vmaxs, vmins = [4, 4, 2], [-1, -1, -2]
        preds = [
            np.triu(self.pred[start:start+700, start:start+700]),
            np.triu(self.pred2[start:start+700, start:start+700]),
            np.triu(self.pred_diff[start:start+700, start:start+700])
        ]
        cts = [self.ct, self.ct2, 'Difference']
        
        for i in range(3):
            ax, pred, ct = axs[i], preds[i], cts[i]
            n = pred.shape[0]
            A = rotate_coord(n)
            img = ax.pcolormesh(A[:, 1].reshape(n+1, n+1), A[:, 0].reshape(n+1, n+1),
                                np.flipud(pred), cmap='RdBu_r', vmax=vmaxs[i], vmin=vmins[i])
            ax.title.set_text(ct)
            ax.set_xlim(300, 500)
            ax.set_ylim(0, 200)
            ax.set_xticklabels(np.linspace(start+300, start+500, num=len(ax.get_xticks()), dtype=int))
            
            x = np.arange(300, 500)
            y1, y2 = np.absolute(2*x-2*locstart), np.absolute(2*x-2*locend)
            ax.plot(x, y1, color='magenta', alpha=0.3)
            ax.plot(x, y2, color='magenta', alpha=0.3)
            plt.colorbar(img, ax=ax)
            
        plt.xlabel('chr{} (10kb)'.format(self.chrom))
        plt.savefig(osp.join(self.fig_dir, 'chr{}_{}_pred.png'.format(self.chrom, gene)))
        plt.clf()
        