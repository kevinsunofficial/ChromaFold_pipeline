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
import warnings

warnings.filterwarnings('ignore')


def plot_ctcf(ctcf, chrom, start, gene, span, fig_path):
    plt.figure(figsize=(8,2))
    plt.plot(ctcf['chr{}'.format(chrom)].toarray()[0][start*200:(start+700)*200])
    plt.xlim(6e4, 1e5)
    plt.ylim(2.8, 3.5)
    plt.title('chr{} - {}'.format(chrom, gene), fontsize=12)
    plt.ylabel('CTCF motif score')
    
    spanstart, spanend = span
    plt.axvspan(spanstart*200, spanend*200, alpha=0.3, color='magenta')
    plt.savefig(osp.join(fig_path, 'chr{}{}_ctcf.png'.format(chrom, gene)))
    plt.clf()
    

def plot_atac(atacs, chrom, cts, start, gene, span, fig_path, control=False):
    plt.rcParams['figure.figsize'] = 8, 5
    plt.rcParams['figure.autolayout'] = False
    
    fig, axs = plt.subplots(3, 1, sharex=True, sharey=True)
    fig.suptitle('chr{} - {}'.format(chrom, gene), fontsize=15)
    fig.tight_layout(rect=[0.01, 0.03, 1, 4.5/5], h_pad=2)
    
    spanstart, spanend = span
    for i in range(3):
        ax, atac, ct = axs[i], atacs[i], cts[i]
        ax.plot(atac)
        ax.title.set_text(ct)
        ax.set_xlim(6e4, 1e5)
        ax.axvspan(spanstart*200, spanend*200, alpha=0.3, color='magenta')
    
    fig.text(-0.03, 0.4, 'ATAC-seq signal', rotation='vertical')
    desc = 'control' if control else 'diff'
    plt.savefig(osp.join(fig_path, 'chr{}{}_atac_{}.png'.format(chrom, gene, desc)))
    plt.clf()
    

def plot_scatac(scatacs, chrom, cts, start, gene, span, fig_path):
    plt.rcParams['figure.figsize'] = 8, 6.5
    plt.rcParams['figure.autolayout'] = False

    fig, axs = plt.subplots(2, 1, sharex=True)
    fig.suptitle('chr{} - {}'.format(chrom, gene), fontsize=15)
    fig.tight_layout(rect=[0.01, 0.03, 1, 6/6.5], h_pad=2)
    
    vmax, vmin = 0.5, 0
    spanstart, spanend = span

    for i in range(2):
        ax, scatac, ct = axs[i], scatacs[i], cts[i]
        C = calc_jaccard(scatac, chrom, start)
        n = C.shape[0]
        t = np.array([[1,0.5],[-1,0.5]])
        A = np.dot(np.array([(i[1],i[0]) for i in itertools.product(range(n,-1,-1),range(0,n+1,1))]),t)
        img = ax.pcolormesh(A[:,1].reshape(n+1,n+1),A[:,0].reshape(n+1,n+1),np.flipud(C),
                            cmap='RdBu_r', vmax=vmax, vmin=vmin)
        ax.title.set_text(ct)
        ax.set_xlim(300,500)
        ax.set_ylim(0,200)
        
        x = np.arange(300,500)
        y1 = np.absolute(2*x-2*spanstart)
        y2 = np.absolute(2*x-2*spanend)
        ax.plot(x,y1,color='magenta',alpha=0.3)
        ax.plot(x,y2,color='magenta',alpha=0.3)
    
    fig.colorbar(img, ax=axs, shrink=0.5)
    tickloc = ax.get_xticks()
    ticklabel = np.linspace((start+300)*10, (start+500)*10, num=len(tickloc),dtype=int).tolist()
    plt.xticks(tickloc, ticklabel)
    plt.savefig(osp.join(fig_path, 'chr{}{}_coacc.png'.format(chrom, gene)))
    plt.clf()
    

def plot_pred(preds, chrom, cts, start, gene, span, fig_path, control=False):
    plt.rcParams['figure.figsize'] = 8, 9.5
    plt.rcParams['figure.autolayout'] = False

    fig, axs = plt.subplots(3, 1, sharex=True)
    fig.suptitle('chr{} - {}'.format(chrom, gene),fontsize=15)
    fig.tight_layout(rect=[0,0.03,1,9/9.5],h_pad=2)
    
    vmax = [4, 4, 4] if control else [4, 4, 2]
    vmin = [-1, -1, -1] if control else [-1, -1, -2]
    spanstart, spanend = span
    
    for i in range(3):
        C, ax, ct = preds[i], axs[i], cts[i]
        n = C.shape[0]
        t = np.array([[1,0.5],[-1,0.5]])
        A = np.dot(np.array([(i[1],i[0]) for i in itertools.product(range(n,-1,-1),range(0,n+1,1))]),t)
        img = ax.pcolormesh(A[:,1].reshape(n+1,n+1),A[:,0].reshape(n+1,n+1),np.flipud(C),
                            cmap='RdBu_r', vmax=vmax[i], vmin=vmin[i])
        fig.colorbar(img,ax=ax)
        ax.title.set_text(ct)
        ax.set_xlim(300,500)
        ax.set_ylim(0,200)
        
        x = np.arange(300,500)
        y1 = np.absolute(2*x-2*spanstart)
        y2 = np.absolute(2*x-2*spanend)
        ax.plot(x,y1,color='magenta',alpha=0.3)
        ax.plot(x,y2,color='magenta',alpha=0.3)
    
    tickloc = ax.get_xticks()
    ticklabel = np.linspace((start+300)*10, (start+500)*10, num=len(tickloc),dtype=int).tolist()
    plt.xticks(tickloc, ticklabel)
    desc = 'control' if control else 'diff'
    plt.savefig(osp.join(fig_path, 'chr{}{}_pred_{}.png'.format(chrom, gene, desc)))
    plt.clf()