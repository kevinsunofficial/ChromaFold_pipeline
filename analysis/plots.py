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


def plot_ctcf(ctcf, chrom, start, gene, locstart, locend, fig_dir):
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.plot(ctcf)
    plt.xlim(300*200, 500*200)
    plt.ylim(2.5, 3.5)
    tickloc = ax.get_xticks()
    ticklabel = np.linspace(start+300, start+500, num=len(tickloc), dtype=int).tolist()
    plt.xticks(tickloc, ticklabel)
    plt.title('chr{} - {}'.format(chrom, gene), fontsize=15)
    plt.xlabel('chr{} (10kb)'.format(chrom))
    plt.ylabel('CTCF motif score')
    plt.axvspan(locstart*200, locend*200, alpha=0.3, color='red')
    plt.savefig(osp.join(fig_dir, 'chr{}_{}_ctcf.png'.format(chrom, gene)))
    plt.clf()

    return
    

def plot_atac(atac, ct, chrom, start, gene, locstart, locend, fig_dir):
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.plot(atac)
    plt.xlim(300*200, 500*200)
    tickloc = ax.get_xticks()
    ticklabel = np.linspace(start+300, start+500, num=len(tickloc), dtype=int).tolist()
    plt.xticks(tickloc, ticklabel)
    plt.title('chr{} - {}'.format(chrom, gene), fontsize=15)
    plt.xlabel('chr{} (10kb)'.format(chrom))
    plt.ylabel('ATAC-seq signal')
    plt.axvspan(locstart*200, locend*200, alpha=0.3, color='red')
    plt.savefig(osp.join(fig_dir, 'chr{}_{}_atac.png'.format(chrom, gene)))
    plt.clf()

    return

    
def plot_scatac(scatac, ct, chrom, start, gene, locstart, locend, fig_dir):
    fig, ax = plt.subplots(figsize=(8, 3))
    vmax, vmin = 0.5, 0
    n = scatac.shape[0]
    A = rotate_coord(n)
    img = ax.pcolormesh(A[:, 1].reshape(n+1, n+1), A[:, 0].reshape(n+1, n+1), 
                        np.flipud(scatac), cmap='RdBu_r', vmax=vmax, vmin=vmin)
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
    plt.title('chr{} - {} ({})'.format(chrom, gene, ct), fontsize=15)
    plt.ylabel('Co-accessibility')
    plt.savefig(osp.join(fig_dir, 'chr{}_{}_scatac.png'.format(chrom, gene)))
    plt.clf()

    return
    

def plot_pred(predmat, ct, chrom, start, gene, locstart, locend, fig_dir):
    fig, ax = plt.subplots(figsize=(8,3))
    vmax, vmin = 4, -1
    pred = np.triu(predmat[start:start+700, start:start+700])
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
    plt.title('chr{} - {} ({})'.format(chrom, gene, ct), fontsize=12)
    plt.ylabel('Prediction Z-score')
    plt.savefig(osp.join(fig_dir, 'chr{}_{}_pred.png'.format(chrom, gene)))
    plt.clf()

    return


def plot_atac_paired(atac1, atac2, ct1, ct2, chrom, start, gene, locstart, locend, fig_dir):
    plt.rcParams['figure.figsize'] = 8, 5
    plt.rcParams['figure.autolayout'] = False
    
    fig, axs = plt.subplots(3, 1, sharex=True, sharey=True)
    fig.suptitle('chr{} - {}'.format(chrom, gene), fontsize=15)
    fig.tight_layout(rect=[0.01, 0.03, 1, 4.5/5], h_pad=2)
    
    atacs = [atac1, atac2, atac1-atac2]
    cts = [ct1, ct2, 'Difference']
    for i in range(3):
        ax, atac, ct = axs[i], atacs[i], cts[i]
        ax.plot(atac)
        ax.title.set_text(ct)
        ax.set_xlim(300*200, 500*200)
        ax.set_xticklabels(np.linspace(start+300, start+500, num=len(ax.get_xticks()), dtype=int))
        ax.axvspan(locstart*200, locend*200, alpha=0.3, color='red')
    
    axs[1].set_ylabel('ATAC-seq signal')
    plt.xlabel('chr{} (10kb)'.format(chrom))
    plt.savefig(osp.join(fig_dir, 'chr{}_{}_atac.png'.format(chrom, gene)))
    plt.clf()

    return


def plot_scatac_paired(scatac1, scatac2, ct1, ct2, chrom, start, gene, locstart, locend, fig_dir):
    plt.rcParams['figure.figsize'] = 8, 6.5
    plt.rcParams['figure.autolayout'] = False

    fig, axs = plt.subplots(2, 1, sharex=True)
    fig.suptitle('chr{} - {}'.format(chrom, gene), fontsize=15)
    fig.tight_layout(rect=[0.01, 0.03, 1, 6/6.5], h_pad=2)
    
    vmax, vmin = 0.5, 0
    scatacs = [scatac1, scatac2]
    cts = [ct1, ct2]
    
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
    plt.xlabel('chr{} (10kb)'.format(chrom))
    plt.savefig(osp.join(fig_dir, 'chr{}_{}_scatac.png'.format(chrom, gene)))
    plt.clf()

    return


def plot_pred_paired(predmat1, predmat2, preddiffmat, ct1, ct2, chrom, start, gene, locstart, locend, fig_dir):
    plt.rcParams['figure.figsize'] = 8, 9.5
    plt.rcParams['figure.autolayout'] = False

    fig, axs = plt.subplots(3, 1, sharex=True)
    fig.suptitle('chr{} - {}'.format(chrom, gene), fontsize=15)
    fig.tight_layout(rect=[0.01, 0.03, 1, 9/9.5], h_pad=2)
    
    vmaxs, vmins = [4, 4, 2], [-1, -1, -2]
    preds = [
        np.triu(predmat1[start:start+700, start:start+700]),
        np.triu(predmat2[start:start+700, start:start+700]),
        np.triu(preddiffmat[start:start+700, start:start+700])
    ]
    cts = [ct1, ct2, 'Difference']
    
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
        
    plt.xlabel('chr{} (10kb)'.format(chrom))
    plt.savefig(osp.join(fig_dir, 'chr{}_{}_pred.png'.format(chrom, gene)))
    plt.clf()

    return


def write_bedpe(coords, chrom, ct, bedpe_dir):
    bedpe = []

    for s1, e1, s2, e2, score in coords:
        line = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
            chrom, int(s1 * 1e4), int(e1 * 1e4), 
            chrom, int(s2 * 1e4), int(e2 * 1e4),
            f'chr{chrom}_gene', score, '.', '.'
        )
        bedpe.append(line)
    
    bedpe_file = osp.join(bedpe_dir, f'{ct}_chr{chrom}.bedpe')
    with open(bedpe_file, 'w') as file:
        file.writelines(bedpe)
    
    return


def plot_track(gtf_file, bedpe_dir, ct, chrom, start, gene, locstart, locend, fig_dir):
    swstr = '0.05+score**2/10'
    hlstart, hlend = int((locstart + start) * 1e4), int((locend + start) * 1e4)
    hlregion = [f'chr{chrom}:{hlstart}-{hlend}']
    gtfs = GTF(gtf_file, length_ratio_thresh=0.005, fontsize=32, height=5) + Title('GTF Annotation')
    arc = Arcs(
        osp.join(bedpe_dir, f'{ct}_chr{chrom}.bedpe'),
        linewidth=None, score_to_width=swstr
    ) + Inverted() + TrackHeight(8) + Title(f'{ct}_chr{chrom}:{gene}')
    hl = HighLights(hlregion, color='red', alpha=0.1)

    frame = XAxis() + gtfs + hl + Spacer(0.5) + arc + hl
    test_range = f'chr{chrom}:{(start+300)*200*50}-{(start+500)*200*50}'
    frame.plot(test_range).savefig(osp.join(fig_dir, f'chr{chrom}_{gene}_bedpe.png'))

    return


def plot_track_paired(gtf_file, bedpe_dir, ct1, ct2, chrom, start, gene, locstart, locend, fig_dir):
    swstr = '0.05+score**2/10'
    hlstart, hlend = int((locstart + start) * 1e4), int((locend + start) * 1e4)
    hlregion = [f'chr{chrom}:{hlstart}-{hlend}']
    gtfs = GTF(gtf_file, length_ratio_thresh=0.005, fontsize=32, height=5) + Title('GTF Annotation')
    arc1 = Arcs(
        osp.join(bedpe_dir, f'{ct1}_chr{chrom}.bedpe'),
        linewidth=None, score_to_width=swstr
    ) + Inverted() + TrackHeight(8) + Title(f'{ct1}_chr{chrom}:{gene}')
    arc2 = Arcs(
        osp.join(bedpe_dir, f'{ct2}_chr{chrom}.bedpe'),
        linewidth=None, score_to_width=swstr
    ) + Inverted() + TrackHeight(8) + Title(f'{ct2}_chr{chrom}:{gene}')
    hl = HighLights(hlregion, color='red', alpha=0.1)

    frame = XAxis() + gtfs + hl + Spacer(0.5) + arc1 + hl + Spacer(0.5) + arc2 + hl
    test_range = f'chr{chrom}:{(start+300)*200*50}-{(start+500)*200*50}'
    frame.plot(test_range).savefig(osp.join(fig_dir, f'chr{chrom}_{gene}_bedpe.png'))

    return


if __name__=='__main__':
    pass
