import os
import os.path as osp
import numpy as np
import pandas as pd
import scipy
from scipy.sparse import csr_matrix
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
import coolbox
from coolbox.api import *
from adjustText import adjust_text
import warnings

warnings.filterwarnings('ignore')


def rotate_coord(n):
    tmp = np.array(list(itertools.product(range(n,-1,-1),range(0,n+1,1))))
    tmp[:, [0, 1]] = tmp[:, [1, 0]]
    A = tmp.dot(np.array([[1, 0.5], [-1, 0.5]]))
    
    return A


class SinglePlot:

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
        self.use_fig_dir = self.fig_dir

    def plot_ctcf(self, start, gene, locstart, locend):
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.plot(self.ctcf[start * 200:(start + 700) * 200])
        plt.xlim(300*200, 500*200)
        plt.ylim(2.5, 3.5)
        tickloc = ax.get_xticks()
        ticklabel = np.linspace(start+300, start+500, num=len(tickloc), dtype=int).tolist()
        plt.xticks(tickloc, ticklabel)
        plt.title('chr{} - {}'.format(self.chrom, gene), fontsize=15)
        plt.xlabel('chr{} (10kb)'.format(self.chrom))
        plt.ylabel('CTCF motif score')
        plt.axvspan(locstart*200, locend*200, alpha=0.3, color='red')
        plt.savefig(osp.join(self.use_fig_dir, 'chr{}_{}_ctcf.png'.format(self.chrom, gene)))
        plt.clf()

    def plot_atac(self, start, gene, locstart, locend):
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.plot(self.atac[start * 200:(start + 700) * 200])
        plt.xlim(300*200, 500*200)
        tickloc = ax.get_xticks()
        ticklabel = np.linspace(start+300, start+500, num=len(tickloc), dtype=int).tolist()
        plt.xticks(tickloc, ticklabel)
        plt.title('chr{} - {} ({})'.format(self.chrom, gene, self.ct), fontsize=15)
        plt.xlabel('chr{} (10kb)'.format(self.chrom))
        plt.ylabel('ATAC-seq signal')
        plt.axvspan(locstart*200, locend*200, alpha=0.3, color='red')
        plt.savefig(osp.join(self.use_fig_dir, 'chr{}_{}_atac.png'.format(self.chrom, gene)))
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
        plt.savefig(osp.join(self.use_fig_dir, 'chr{}_{}_scatac.png'.format(self.chrom, gene)))
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
        plt.savefig(osp.join(self.use_fig_dir, 'chr{}_{}_pred.png'.format(self.chrom, gene)))
        plt.clf()

    def plot_track(self, start, gene, locstart, locend):
        swstr = '0.01+score**2/50'
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
        frame.plot(test_range).savefig(osp.join(self.use_fig_dir, f'{self.ct}_chr{self.chrom}_{gene}_bedpe.png'))
        plt.clf()


class PairPlot(SinglePlot):

    def __init__(self, fig_dir, bedpe_dir, gtf_file, chrom, 
                 ct, ct2, pred, pred2, pred_diff,
                 ctcf=None, atac=None, scatac=None,
                 atac2=None, scatac2=None):
        super().__init__(
            fig_dir, bedpe_dir, gtf_file, chrom, 
            ct, pred, ctcf, atac, scatac)
        self.ct2 = ct2
        self.pred2 = pred2
        self.pred_diff = pred_diff
        self.atac2 = atac2
        self.scatac2 = scatac2
        self.use_fig_dir = self.fig_dir

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
            ax.plot(atac[start * 200:(start + 700) * 200])
            ax.title.set_text(ct)
            ax.set_xlim(300*200, 500*200)
            ax.set_xticklabels(np.linspace(start+300, start+500, num=len(ax.get_xticks()), dtype=int))
            ax.axvspan(locstart*200, locend*200, alpha=0.3, color='red')
        
        axs[1].set_ylabel('ATAC-seq signal')
        plt.xlabel('chr{} (10kb)'.format(self.chrom))
        plt.savefig(osp.join(self.use_fig_dir, 'chr{}_{}_atac.png'.format(self.chrom, gene)))
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
        plt.savefig(osp.join(self.use_fig_dir, 'chr{}_{}_scatac.png'.format(self.chrom, gene)))
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
        plt.savefig(osp.join(self.use_fig_dir, 'chr{}_{}_pred.png'.format(self.chrom, gene)))
        plt.clf()

    def plot_track(self, start, gene, locstart, locend):
        swstr = '0.01+score**2/50'
        hlstart, hlend = int((locstart + start) * 1e4), int((locend + start) * 1e4)
        hlregion = [f'chr{self.chrom}:{hlstart}-{hlend}']
        gtfs = GTF(self.gtf_file, length_ratio_thresh=0.005, fontsize=32, height=5) + Title('GTF Annotation')
        arc1 = Arcs(
            osp.join(self.bedpe_dir, f'{self.ct}_chr{self.chrom}.bedpe'),
            linewidth=None, score_to_width=swstr
        ) + Inverted() + TrackHeight(8) + Title(f'{self.ct}_chr{self.chrom}:{gene}')
        arc2 = Arcs(
            osp.join(self.bedpe_dir, f'{self.ct2}_chr{self.chrom}.bedpe'),
            linewidth=None, score_to_width=swstr
        ) + Inverted() + TrackHeight(8) + Title(f'{self.ct2}_chr{self.chrom}:{gene}')
        hl = HighLights(hlregion, color='red', alpha=0.1)

        frame = XAxis() + gtfs + hl + Spacer(0.5) + arc1 + hl + Spacer(0.5) + arc2 + hl
        test_range = f'chr{self.chrom}:{(start+300)*200*50}-{(start+500)*200*50}'
        frame.plot(test_range).savefig(osp.join(self.use_fig_dir, f'chr{self.chrom}_{gene}_bedpe.png'))


class PairGenePlotGenerator(PairPlot):

    def __init__(self, gene_score, fig_dir, bedpe_dir, gtf_file, chrom, 
                 ct, ct2, pred, pred2, pred_diff,
                 ctcf=None, atac=None, scatac=None,
                 atac2=None, scatac2=None):
        super().__init__(
            fig_dir, bedpe_dir, gtf_file, chrom, 
            ct, ct2, pred, pred2, pred_diff,
            ctcf, atac, scatac, atac2, scatac2)
        self.gene_score = gene_score
        self.use_fig_dir = self.fig_dir

    def plot_gene(self, index=None, gene=None):
        assert index is not None or gene is not None, \
            'Either index or gene name is required, both are missing'
        if index is None:
            indices = self.gene_score.index[self.gene_score.gene_name==gene].tolist()
            assert indices, f'{gene} does not exist in query result'
            index = indices[0]
        if gene is None:
            gene = self.gene_score.iloc[index].gene_name
        actual_start = self.gene_score.iloc[index].start
        actual_end = self.gene_score.iloc[index].end
        rank = index + 1
        self.use_fig_dir = osp.join(self.fig_dir, f'{rank}_{gene}')
        if not osp.exists(self.use_fig_dir):
            os.makedirs(self.use_fig_dir)
        
        start = max(0, actual_start // 10**4 - 400)
        locstart, locend = actual_start / 1e4 - start, actual_end / 1e4 - start

        if self.ctcf is not None:
            super().plot_ctcf(start, gene, locstart, locend)
        if self.atac is not None and self.atac2 is not None:
            super().plot_atac(start, gene, locstart, locend)
        if self.scatac is not None and self.scatac2 is not None:
            raise NotImplementedError

        super().plot_pred(start, gene, locstart, locend)
        super().plot_track(start, gene, locstart, locend)
    
    def plot_genes(self, numplot):
        usenum = min(numplot, self.gene_score.shape[0])
        for i in tqdm(range(usenum)):
            self.plot_gene(index=i)

    def volcano_plot(self, t, m, text=False):
        threshold = 15 if t == 'ranksums' else 30
        pvals = self.gene_score[(self.gene_score[t] >= threshold) & (self.gene_score[m] >= .15)]
        means = self.gene_score[self.gene_score[m] >= .25]
        allsigs = pd.merge(pvals, means, how='outer')
        mostsigs = pd.merge(pvals, means, how='inner')

        plt.figure(figsize=(6, 4))
        plt.scatter(self.gene_score[m], self.gene_score[t], alpha=0.2, 
                    s=3, color='grey', label='not significant')
        plt.scatter(pvals[m], pvals[t],
                    s=5, color='red', label='-log10(p_value)')
        plt.scatter(means[m], means[t],
                    s=5, color='blue', label=f'average {m}')
        plt.scatter(mostsigs[m], mostsigs[t],
                    s=5, color='purple', label='both')
        
        if text:
            texts = [
                plt.text(r[m], r[t], r['gene_name'], fontsize=9) for i, r in allsigs.iterrows()
            ]
            adjust_text(texts, arrowprops=dict(arrowstyle='-', color='black', lw=.5))
        
        plt.xlabel(f'Mean of {m} values')
        plt.ylabel(f'-log10 p_value for {t} test')
        plt.axhline(threshold, color='grey', linestyle='--', lw=1)
        plt.legend()
        plt.savefig(self.fig_dir, f'chr{self.chrom}_genes_volcano_plot.png'))
        plt.clf()


class PairTADPlotGenerator(PairPlot):

    def __init__(self, tad_score, fig_dir, bedpe_dir, gtf_file, chrom, 
                 ct, ct2, pred, pred2, pred_diff,
                 ctcf=None, atac=None, scatac=None,
                 atac2=None, scatac2=None):
        super().__init__(
            fig_dir, bedpe_dir, gtf_file, chrom, 
            ct, ct2, pred, pred2, pred_diff,
            ctcf, atac, scatac, atac2, scatac2)
        self.tad_score = tad_score
        self.use_fig_dir = self.fig_dir

    def plot_tad(self, index):
        actual_start = int(self.tad_score.iloc[index].start)
        actual_end = int(self.tad_score.iloc[index].end)
        midplace = (actual_end + actual_start) // 2
        start = max(0, midplace - 400)
        locstart, locend = actual_start - start, actual_end - start
        gene = f'{actual_start}_{actual_end}'

        rank = index + 1
        self.use_fig_dir = osp.join(self.fig_dir, f'{rank}_{gene}')
        if not osp.exists(self.use_fig_dir):
            os.makedirs(self.use_fig_dir)

        if self.ctcf is not None:
            super().plot_ctcf(start, gene, locstart, locend)
        if self.atac is not None and self.atac2 is not None:
            super().plot_atac(start, gene, locstart, locend)
        if self.scatac is not None and self.scatac2 is not None:
            raise NotImplementedError

        super().plot_pred(start, gene, locstart, locend)
        super().plot_track(start, gene, 0, 200)
    
    def plot_tads(self, numplot):
        usenum = min(numplot, self.tad_score.shape[0])
        for i in tqdm(range(usenum)):
            self.plot_tad(index=i)

    def volcano_plot(self, t, m, text=False):
        threshold = 15 if t == 'ranksums' else 30
        pvals = self.tad_score[(self.tad_score[t] >= threshold) & (self.tad_score[m] >= .15)]
        means = self.tad_score[self.tad_score[m] >= .25]
        allsigs = pd.merge(pvals, means, how='outer')
        mostsigs = pd.merge(pvals, means, how='inner')

        plt.figure(figsize=(6, 4))
        plt.scatter(self.tad_score[m], self.tad_score[t], alpha=0.2, 
                    s=3, color='grey', label='not significant')
        plt.scatter(pvals[m], pvals[t],
                    s=5, color='red', label='-log10(p_value)')
        plt.scatter(means[m], means[t],
                    s=5, color='blue', label=f'average {m}')
        plt.scatter(mostsigs[m], mostsigs[t],
                    s=5, color='purple', label='both')
        
        if text:
            texts = [
                plt.text(r[m], r[t], f'{r['start']}_{r['end']}', fontsize=9) for i, r in allsigs.iterrows()
            ]
            adjust_text(texts, arrowprops=dict(arrowstyle='-', color='black', lw=.5))
        
        plt.xlabel(f'Mean of {m} values')
        plt.ylabel(f'-log10 p_value for {t} test')
        plt.axhline(threshold, color='grey', linestyle='--', lw=1)
        plt.legend()
        plt.savefig(self.fig_dir, f'chr{self.chrom}_TADs_volcano_plot.png'))
        plt.clf()
