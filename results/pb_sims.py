#!/usr/bin/python
#$: chmod 755 yourfile.py
#$: dos2unix yourfile.py
#$: ./yourfile.py

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.axis
from matplotlib import rcParams
import time
import scipy
import math
from uncertainties import ufloat
import sys

np.set_printoptions(threshold=np.inf)

figx = 10
figy = 8
font_size = 12
rcParams['axes.labelsize'] = font_size
rcParams['xtick.labelsize'] = font_size
rcParams['ytick.labelsize'] = font_size
rcParams['legend.fontsize'] = font_size
rcParams['figure.figsize'] = figx, figy




def myRg(N):
    Rout = 0.4348*(N)**0.61
    return Rout
def myRho_c(Rg):
    return (1)/((np.pi)*Rg**2)
def myFindLine(filename,keywords):
    with open(filename, 'r') as f:
        fLine = []
        for n, line in enumerate(f):
            checks = [True if (keywords[i] in line) else False for i in range(len(keywords))]
            if all(checks):
                fLine.append(n)
    return fLine
def myLastLine(filename):
    with open(filename, 'rb') as f:
        try:  # catch OSError in case of a one line file
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b'\n':
                f.seek(-2, os.SEEK_CUR)
        except OSError:
            f.seek(0)
        last_line = f.readline().decode()
    return last_line

def myFindLines(filename,tars):
    # filename: txt file
    # tars: list of lists of keywords per line
    lines = [myFindLine(filename,tars[i]) for i in range(len(tars))]
    return(lines)
def myReadLines(filename, rlines):
    lines = []
    with open(filename, 'r') as f:
        for n, line in enumerate(f):
            if n in rlines:
                lines.append(line)
        f.close()
    return lines

def myCompletedVs(PBBname):
    CompletedVs = [os.path.join(PBBname, o) for o in os.listdir(PBBname) if os.path.isdir(os.path.join(PBBname, o))]
    return CompletedVs

def reorder_data(data,order=None):
    if order is None:
        order = [i for i in range(len(data.shape))]
    data_r = np.moveaxis(data, [i for i in range(len(data.shape))], order)
    return data_r
def reorder_datas(datas, order):
    for d in range(len(datas)):
        while len(datas[d].shape) < 4:
            datas[d] = np.expand_dims(datas[d], axis=0)
        if order is None:
            order = [i for i in range(len(datas[d].shape))]
        datas[d] = np.moveaxis(datas[d], [i for i in range(len(datas[d].shape))], order)
    return None

def Dint(PvD,Ps):
    dims = (PvD.shape[0], PvD.shape[1], PvD.shape[2], len(Ps), 2)
    Dint=np.zeros(dims)
    for Ni in range(PvD.shape[0]):
        for Mi in range(PvD.shape[1]):
            for Xi in  range(PvD.shape[2]):
                Pf = np.flip(PvD[Ni,Mi,Xi,:,1,0])
                Df = np.flip(PvD[Ni,Mi,Xi,:,0,0])
                Di = np.interp(Ps,Pf,Df,right=np.NaN,left=np.NaN)
                Di2 = np.expand_dims(Di,axis=-1)
                Ps_array = np.asarray(Ps)
                Ps2 = np.expand_dims(Ps_array,axis=-1)
                Dint[Ni, Mi, Xi] = np.concatenate((Ps2,Di2),axis=-1)
    return Dint


def D2P(A,Ds,Dint,Ps):

    dims = (A.shape[0], A.shape[1], A.shape[2], Dint.shape[3], A.shape[4], A.shape[5], A.shape[6])
    print(dims)
    B = np.zeros(dims)
    for Ni in range(A.shape[0]):
        for Mi in range(A.shape[1]):
            for Xi in range(A.shape[2]):
                for Vi in range(A.shape[4]):
                    for Tvari in range(A.shape[5]):

                        Tvar_int = np.interp(Dint[Ni,Mi,Xi,:,1],Ds,A[Ni,Mi,Xi,:,Vi,Tvari,0])
                        avg_error = np.mean(A[Ni,Mi,Xi,:,Vi,Tvari,1])

                        if np.ma.is_masked(avg_error):
                            avg_error=np.NaN
                        new_errors = [np.NaN if np.isnan(val) else avg_error for val in Tvar_int]
                        B[Ni,Mi,Xi,:,Vi,Tvari,0] = Tvar_int
                        B[Ni, Mi, Xi, :, Vi, Tvari, 1] = new_errors
    return np.ma.masked_invalid(B)

def myPlot(datas,styles=['-','--','-.',':'],order=None,llabels=(None,None),xlabel=None,ylabel=None,title=None,fname='plot',
           names=None,xlim=False,axis2=None,y2label=None,log=(False,False)):
    lloc = llabels[0]
    labels = llabels[1]
    for d in range(len(datas)):
        while len(datas[d].shape) < 4:
            datas[d] = np.expand_dims(datas[d], axis=0)
        if order is None:
            order = [i for i in range(len(datas[d].shape))]
        datas[d] = np.moveaxis(datas[d], [i for i in range(len(datas[d].shape))], order)
    for i in range(datas[0].shape[0]):
        fig, ax = plt.subplots()
        if axis2 != None:
            ax2 = ax.twinx()
        l = 0
        d = 0
        for data in datas:
            if axis2 is None:
                ax1ax2 = [0] * round((data.shape[2])/2)
            else:
                ax1ax2 = axis2.copy()
            nlines = data.shape[1] * (data.shape[2] / 2)
            if nlines > 4:
                colors = ['r', 'y', 'g', 'c', 'b', 'm', 'k'] * 5
            elif nlines == 1:
                colors = ['k']
            else:
                colors = ['r', 'g', 'b', 'k'] * 10
            c = 0
            for j in range(data.shape[1]):
                for k in range(round((data.shape[2])/2)):
                    form = '{0}{1}'.format(styles[d],colors[c])
                    if ax1ax2[k]==0:
                        if(lloc==0 and j==0 and k==[0 if round((data.shape[2])/2)<2 else 1][0] ):
                            ax.plot(data[i, j, k * 2, :].compressed(), data[i, j, (k * 2) + 1, :].compressed(), form,label=labels[l] )
                            l+=1
                        elif(lloc==1 and d==0 and k==[0 if round((data.shape[2])/2)<2 else 1][0]):
                            ax.plot(data[i, j, k * 2, :].compressed(), data[i, j, (k * 2) + 1, :].compressed(), form,label=labels[l] )
                            l+=1
                        elif(lloc==2 and d==0 and j==0):
                            ax.plot(data[i, j, k * 2, :].compressed(), data[i, j, (k * 2) + 1, :].compressed(), form,label=labels[l] )
                            l+=1
                        else:
                            ax.plot(data[i, j, k * 2, :].compressed(), data[i, j, (k * 2) + 1, :].compressed(), form)
                    else:
                        if (lloc == 0 and j == 0 and k == [0 if round((data.shape[2]) / 2) < 2 else 1][0]):
                            ax2.plot(data[i, j, k * 2, :].compressed(), data[i, j, (k * 2) + 1, :].compressed(), form,
                                    label=labels[l])
                            l += 1
                        elif (lloc == 1 and d == 0 and k==[0 if round((data.shape[2])/2)<2 else 1][0]):
                            ax2.plot(data[i, j, k * 2, :].compressed(), data[i, j, (k * 2) + 1, :].compressed(), form,
                                    label=labels[l])
                            l += 1
                        elif (lloc == 2 and d == 0 and j == 0):
                            ax2.plot(data[i, j, k * 2, :].compressed(), data[i, j, (k * 2) + 1, :].compressed(), form,
                                    label=labels[l])
                            l += 1
                        else:
                            ax2.plot(data[i, j, k * 2, :].compressed(), data[i, j, (k * 2) + 1, :].compressed(), form)

                    c+=1
            d+=1
        ax.set(xlim=xlim, xlabel=xlabel, ylabel=ylabel, title=title)
        if log[0] is True:
            plt.xscale('log')
        if log[1] is True:
            plt.yscale('log')
        if axis2 != None:
            ax.legend(loc=8)
            ax2.legend(loc=4)
            ax2.minorticks_on()
            ax2.set(ylabel=y2label)
        else:
            ax.legend(loc=0)
        ax.minorticks_on()
        ax.grid(visible=True, which='both', axis='both')
        # plt.margins(0,0)
        if names is None:
            figurename = '{0}-{1}.jpg'.format(fname, i)
        else:
            figurename = '{0}-{1}.jpg'.format(fname, names[i])
        print(figurename)
        plt.savefig(figurename, bbox_inches='tight')
        plt.close()
    return None
def myErrorPlot(datas,styles=['o-','o--','o-.','o:'],order=None,bars=True,llabels=(None,None),xlabel=None,ylabel=None,title=None,fname='plot',
                names=None,xlim=False,axis2=None,y2label=None,styles2=None,log=(False,False),power_law=(0,0,'y','x')):
    lloc = llabels[0]
    labels = llabels[1]
    for d in range(len(datas)):
        while len(datas[d].shape) < 5:
            datas[d] = np.expand_dims(datas[d], axis=0)
        if order is None:
            order = [i for i in range(len(datas[d].shape))]
        datas[d] = np.moveaxis(datas[d], [i for i in range(len(datas[d].shape))], order)
    if styles2 is None:
        styles2 = [x.replace('o','^') for x in styles]
    for i in range(datas[0].shape[0]):
        fig, ax = plt.subplots()
        if axis2 != None:
            ax2 = ax.twinx()
        l = 0
        d = 0
        for data in datas:
            if axis2 is None:
                ax1ax2 = [0] * round((data.shape[2])/2)
            else:
                ax1ax2 = axis2.copy()
            nlines = data.shape[1] * (data.shape[3] / 2)
            if nlines > 4:
                colors = ['r', 'y', 'g', 'c', 'b', 'm', 'k'] * 5
            elif nlines == 1:
                colors = ['k']
            else:
                colors = ['r', 'g', 'b', 'k'] * 10
            c1 = 0
            c2 = 0
            for j in range(data.shape[1]):
                for k in range(round((data.shape[3])/2)):
                    xs = data[i,j,:,k * 2,0].compressed()
                    ys = data[i,j,:,(k * 2) + 1, 0].compressed()
                    xerrs = data[i, j,:, k * 2, 1].compressed()
                    yerrs = data[i, j,:, (k * 2)+1, 1].compressed()
                    if ax1ax2[k]==0:
                        form = '{0}{1}'.format(styles[d], colors[c1])
                        if(lloc==0 and j==0 and k==[0 if round((data.shape[3])/2)<2 else 1][0] ):
                            if bars is True:
                                ax.errorbar(xs, ys, fmt=form, xerr=xerrs, yerr=yerrs, capsize=5, label=labels[l])
                            else:
                                ax.errorbar(xs, ys, fmt=form, capsize=5, label=labels[l])
                            l+=1
                        elif(lloc==1 and d==0 and k==[0 if round((data.shape[3])/2)<2 else 1][0]):
                            if bars is True:
                                ax.errorbar(xs, ys, fmt=form, xerr=xerrs, yerr=yerrs, capsize=5, label=labels[l])
                            else:
                                ax.errorbar(xs, ys, fmt=form, capsize=5, label=labels[l])
                            l+=1
                        elif(lloc==2 and d==0 and j==0):
                            if bars is True:
                                ax.errorbar(xs, ys, fmt=form, xerr=xerrs, yerr=yerrs, capsize=5, label=labels[l])
                            else:
                                ax.errorbar(xs, ys, fmt=form, capsize=5, label=labels[l])
                            l+=1
                        else:
                            if bars is True:
                                ax.errorbar(xs, ys, fmt=form, xerr=xerrs, yerr=yerrs, capsize=5)
                            else:
                                ax.errorbar(xs, ys, fmt=form, capsize=5)
                        c1 += 1

                    else:
                        form = '{0}{1}'.format(styles2[d], colors[c2])
                        if (lloc == 0 and j == 0 and k == [0 if round((data.shape[3]) / 2) < 2 else 1][0]):
                            if bars is True:
                                ax2.errorbar(xs, ys, fmt=form, xerr=xerrs, yerr=yerrs, capsize=5, label=labels[l])
                            else:
                                ax2.errorbar(xs, ys, fmt=form, capsize=5, label=labels[l])
                            l += 1
                        elif (lloc == 1 and d == 0 and k==[0 if round((data.shape[3])/2)<2 else 1][0]):
                            if bars is True:
                                ax2.errorbar(xs, ys, fmt=form, xerr=xerrs, yerr=yerrs, capsize=5, label=labels[l])
                            else:
                                ax2.errorbar(xs, ys, fmt=form, capsize=5, label=labels[l])
                            l += 1

                        elif (lloc == 2 and d == 0 and j == 0):
                            if bars is True:
                                ax2.errorbar(xs, ys, fmt=form, xerr=xerrs, yerr=yerrs, capsize=5, label=labels[l])
                            else:
                                ax2.errorbar(xs, ys, fmt=form, capsize=5, label=labels[l])
                            l += 1
                        else:
                            if bars is True:
                                ax2.errorbar(xs, ys, fmt=form, xerr=xerrs, yerr=yerrs, capsize=5)
                            else:
                                ax2.errorbar(xs, ys, fmt=form, capsize=5)
                        c2 += 1
            d+=1
        if power_law[0] != 0:
            xbounds = ax.get_xbound()
            x = np.linspace(xbounds[0] + 0.1 * (xbounds[1] - xbounds[0]), xbounds[0] + 0.66 * (xbounds[1] - xbounds[0]),50)
            y = power_law[0] * x ** power_law[1]
            ax.plot(x, y, 'k--', label=r'$%s \sim %s^{%.2f}$' % (power_law[2], power_law[3], power_law[1]))
        if log[0] is True:
            ax.set_xscale('log')
        if log[1] is True:
            ax.set_yscale('log')
            if axis2 != None:
                ax2.set_yscale('log')

        if (log[0] is True) or (log[1] is True):
            ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        else:
            ax.set(xlim=xlim, xlabel=xlabel, ylabel=ylabel, title=title)

        if axis2 != None:
            ax.legend(loc=8)
            ax2.legend(loc=4)
            ax2.minorticks_on()
            ax2.set(ylabel=y2label)
        else:
            ax.legend(loc=0)
        ax.minorticks_on()
        ax.grid(visible=True, which='both', axis='both')
        # plt.margins(0,0)
        if names is None:
            figurename = '{0}-{1}.jpg'.format(fname, i)
        else:
            figurename = '{0}-{1}.jpg'.format(fname, names[i])
        print(figurename)
        plt.savefig(figurename, bbox_inches='tight')
        plt.close()
    return None
def myErrorPlot_nocomp(datas,styles=['o-','o--','o-.','o:'],order=None,bars=True,llabels=(None,None),xlabel=None,ylabel=None,title=None,fname='plot',
                names=None,xlim=False,axis2=None,y2label=None,styles2=None,log=(False,False),power_law=(0,0,'y','x')):
    lloc = llabels[0]
    labels = llabels[1]
    for d in range(len(datas)):
        while len(datas[d].shape) < 5:
            datas[d] = np.expand_dims(datas[d], axis=0)
        if order is None:
            order = [i for i in range(len(datas[d].shape))]
        datas[d] = np.moveaxis(datas[d], [i for i in range(len(datas[d].shape))], order)
    if styles2 is None:
        styles2 = [x.replace('o','^') for x in styles]
    for i in range(datas[0].shape[0]):
        fig, ax = plt.subplots()
        if axis2 != None:
            ax2 = ax.twinx()
        l = 0
        d = 0
        for data in datas:
            if axis2 is None:
                ax1ax2 = [0] * round((data.shape[2])/2)
            else:
                ax1ax2 = axis2.copy()
            nlines = data.shape[1] * (data.shape[3] / 2)
            if nlines > 4:
                colors = ['r', 'y', 'g', 'c', 'b', 'm', 'k'] * 5
            elif nlines == 1:
                colors = ['k']
            else:
                colors = ['r', 'g', 'b', 'k'] * 10
            c1 = 0
            c2 = 0
            for j in range(data.shape[1]):
                for k in range(round((data.shape[3])/2)):
                    xs = data[i,j,:,k * 2,0]#.compressed()
                    ys = data[i,j,:,(k * 2) + 1, 0]#.compressed()
                    xerrs = data[i, j,:, k * 2, 1]#.compressed()
                    yerrs = data[i, j,:, (k * 2)+1, 1]#.compressed()
                    if ax1ax2[k]==0:
                        form = '{0}{1}'.format(styles[d], colors[c1])
                        if(lloc==0 and j==0 and k==[0 if round((data.shape[3])/2)<2 else 1][0] ):
                            if bars is True:
                                ax.errorbar(xs, ys, fmt=form, xerr=xerrs, yerr=yerrs, capsize=5, label=labels[l])
                            else:
                                ax.errorbar(xs, ys, fmt=form, capsize=5, label=labels[l])
                            l+=1
                        elif(lloc==1 and d==0 and k==[0 if round((data.shape[3])/2)<2 else 1][0]):
                            if bars is True:
                                ax.errorbar(xs, ys, fmt=form, xerr=xerrs, yerr=yerrs, capsize=5, label=labels[l])
                            else:
                                ax.errorbar(xs, ys, fmt=form, capsize=5, label=labels[l])
                            l+=1
                        elif(lloc==2 and d==0 and j==0):
                            if bars is True:
                                ax.errorbar(xs, ys, fmt=form, xerr=xerrs, yerr=yerrs, capsize=5, label=labels[l])
                            else:
                                ax.errorbar(xs, ys, fmt=form, capsize=5, label=labels[l])
                            l+=1
                        else:
                            if bars is True:
                                ax.errorbar(xs, ys, fmt=form, xerr=xerrs, yerr=yerrs, capsize=5)
                            else:
                                ax.errorbar(xs, ys, fmt=form, capsize=5)
                        c1 += 1

                    else:
                        form = '{0}{1}'.format(styles2[d], colors[c2])
                        if (lloc == 0 and j == 0 and k == [0 if round((data.shape[3]) / 2) < 2 else 1][0]):
                            if bars is True:
                                ax2.errorbar(xs, ys, fmt=form, xerr=xerrs, yerr=yerrs, capsize=5, label=labels[l])
                            else:
                                ax2.errorbar(xs, ys, fmt=form, capsize=5, label=labels[l])
                            l += 1
                        elif (lloc == 1 and d == 0 and k==[0 if round((data.shape[3])/2)<2 else 1][0]):
                            if bars is True:
                                ax2.errorbar(xs, ys, fmt=form, xerr=xerrs, yerr=yerrs, capsize=5, label=labels[l])
                            else:
                                ax2.errorbar(xs, ys, fmt=form, capsize=5, label=labels[l])
                            l += 1

                        elif (lloc == 2 and d == 0 and j == 0):
                            if bars is True:
                                ax2.errorbar(xs, ys, fmt=form, xerr=xerrs, yerr=yerrs, capsize=5, label=labels[l])
                            else:
                                ax2.errorbar(xs, ys, fmt=form, capsize=5, label=labels[l])
                            l += 1
                        else:
                            if bars is True:
                                ax2.errorbar(xs, ys, fmt=form, xerr=xerrs, yerr=yerrs, capsize=5)
                            else:
                                ax2.errorbar(xs, ys, fmt=form, capsize=5)
                        c2 += 1
            d+=1
        if power_law[0] != 0:
            xbounds = ax.get_xbound()
            x = np.linspace(xbounds[0] + 0.1 * (xbounds[1] - xbounds[0]), xbounds[0] + 0.66 * (xbounds[1] - xbounds[0]),50)
            y = power_law[0] * x ** power_law[1]
            ax.plot(x, y, 'k--', label=r'$%s \sim %s^{%.2f}$' % (power_law[2], power_law[3], power_law[1]))
        if log[0] is True:
            ax.set_xscale('log')
        if log[1] is True:
            ax.set_yscale('log')
            if axis2 != None:
                ax2.set_yscale('log')

        if (log[0] is True) or (log[1] is True):
            ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        else:
            ax.set(xlim=xlim, xlabel=xlabel, ylabel=ylabel, title=title)

        if axis2 != None:
            ax.legend(loc=8)
            ax2.legend(loc=4)
            ax2.minorticks_on()
            ax2.set(ylabel=y2label)
        else:
            ax.legend(loc=0)
        ax.minorticks_on()
        ax.grid(visible=True, which='both', axis='both')
        # plt.margins(0,0)
        if names is None:
            figurename = '{0}-{1}.jpg'.format(fname, i)
        else:
            figurename = '{0}-{1}.jpg'.format(fname, names[i])
        print(figurename)
        plt.savefig(figurename, bbox_inches='tight')
        plt.close()
    return None

def clear_twin(ax):
    for other_ax in ax.figure.axes:
        if other_ax is ax:
            continue
        if other_ax.bbox.bounds == ax.bbox.bounds:
            other_ax.clear()
            # other_ax.xaxis.set_ticks([])
            # other_ax.xaxis.set_ticklabels([])
            other_ax.yaxis.set_ticks([])
            other_ax.yaxis.set_ticklabels([])
    return False

def has_twin(ax):
    for other_ax in ax.figure.axes:
        if other_ax is ax:
            continue
        if other_ax.bbox.bounds == ax.bbox.bounds:
            return True
    return False

def myAxs(datas,ax,linestyles=['solid','dashed','dashdot','dotted'],markers=['o','^'],llabels=(None,None),xlabel=None,ylabel=None,title=None,xlim=False,axis2=None,y2label=None,log=(False,False), scale=(1,1), theme=None):
    lloc = llabels[0]
    labels = llabels[1]
    for d in range(len(datas)):
        while len(datas[d].shape) < 3:
            datas[d] = np.expand_dims(datas[d], axis=0)
    if has_twin(ax):
        clear_twin(ax)
        # matplotlib.axis.YAxis.reset_ticks
    if axis2 != None:
        ax2 = ax.twinx()
    l = 0
    d = 0
    axs = []
    for data in datas:
        if axis2 is None:
            ax1ax2 = [0] * math.ceil((data.shape[2]) / 2)
        else:
            ax1ax2 = axis2.copy()
        nlines = data.shape[0] * (data.shape[1] / 2)
        if theme == 'rgbb':
            colors = ['red', 'green', 'blue', 'black'] * 10
        elif theme == 'dps':
            colors = ['green', 'yellow', 'cyan', 'black'] * 10
        elif theme == 'edps':
            colors = ['green', 'cyan', 'black'] * 10
        else:
            if nlines > 4:
                colors = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'violet', 'purple', 'maroon', 'black',
                          'gold', 'olive', 'teal'] * 5
            elif nlines == 1:
                colors = ['black']
            else:
                colors = ['red', 'green', 'blue', 'black'] * 10
        c = 0
        for j in range(data.shape[0]):
            for k in range(math.ceil((data.shape[1]) / 2)):
                xs = data[j, k * 2, :].compressed() * scale[0]
                ys = data[j, (k * 2) + 1, :].compressed() * scale[1]
                if ax1ax2[k] == 0:
                    if (lloc == 0 and j == 0 and k == [0 if round((data.shape[1]) / 2) < 2 else 1][0]):
                        ax.plot(xs, ys, color=colors[c], linestyle=linestyles[d], label=labels[l])
                        l += 1
                    elif (lloc == 1 and d == 0 and k == [0 if round((data.shape[1]) / 2) < 2 else 1][0]):
                        ax.plot(xs, ys, color=colors[c], linestyle=linestyles[d], label=labels[l])
                        l += 1
                    elif (lloc == 2 and d == 0 and j == 0):
                        ax.plot(xs, ys, color=colors[c], linestyle=linestyles[d], label=labels[l])
                        l += 1
                    else:
                        ax.plot(xs, ys, color=colors[c], linestyle=linestyles[d])
                else:
                    if (lloc == 0 and j == 0 and k == [0 if round((data.shape[1]) / 2) < 2 else 1][0]):
                        ax2.plot(xs, ys, color=colors[c], linestyle=linestyles[d], label=labels[l])
                        l += 1
                    elif (lloc == 1 and d == 0 and k == [0 if round((data.shape[1]) / 2) < 2 else 1][0]):
                        ax2.plot(xs, ys, color=colors[c], linestyle=linestyles[d], label=labels[l])
                        l += 1
                    elif (lloc == 2 and d == 0 and j == 0):
                        ax2.plot(xs, ys, color=colors[c], linestyle=linestyles[d], label=labels[l])
                        l += 1
                    else:
                        ax2.plot(xs, ys, color=colors[c], linestyle=linestyles[d])
                c += 1
        d += 1
    ax.set(xlim=xlim, xlabel=xlabel, ylabel=ylabel, title=title)
    if log[0] is True:
        plt.xscale('log')
    if log[1] is True:
        plt.yscale('log')
    if axis2 != None:
        ax.legend(loc=8)
        ax2.legend(loc=4)
        ax2.minorticks_on()
        ax2.set(ylabel=y2label)
    else:
        ax.legend(loc=0)
    ax.minorticks_on()
    ax.grid(visible=True, which='both', axis='both')
    axs.append(ax)
    if axis2 != None:
        axs.append(ax2)
    return axs
def myErrorAxs(datas,ax,linestyles=['solid','dashed','dashdot','dotted'],markers=['o','^'],bars=True,llabels=(None,None),xlabel=None,ylabel=None,title=None,xlim=False,axis2=None,y2label=None,styles2=None,log=(False,False),power_law=(0,0,'y','x'),scale=(1,1)):
    lloc = llabels[0]
    labels = llabels[1]
    for d in range(len(datas)):
        while len(datas[d].shape) < 4:
            datas[d] = np.expand_dims(datas[d], axis=0)
    # if styles2 is None:
    #     styles2 = [x.replace('o','^') for x in styles]
    if has_twin(ax):
        clear_twin(ax)
    if axis2 != None:
        ax2 = ax.twinx()
    l = 0
    d = 0
    axs = []
    for data in datas:
        if axis2 is None:
            ax1ax2 = [0] * round((data.shape[1]) / 2)
        else:
            ax1ax2 = axis2.copy()
        nlines = data.shape[0] * (data.shape[2] / 2)
        if nlines > 4:
            colors = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'violet', 'purple', 'maroon', 'black', 'gold',
                      'olive', 'teal', ] * 5
        elif nlines == 1:
            colors = ['black']
        else:
            colors = ['red', 'green', 'blue', 'black'] * 10
        c1 = 0
        c2 = 0
        for j in range(data.shape[0]):
            for k in range(round((data.shape[2]) / 2)):
                xs0 = data[j, :, k * 2, 0]
                ys0 = data[j, :, (k * 2) + 1, 0]
                xerrs0 = data[j, :, k * 2, 1]
                yerrs0 = data[j, :, (k * 2) + 1, 1]
                if np.ma.count_masked(xs0) > np.ma.count_masked(ys0):
                    ys0 = np.ma.masked_where(np.ma.getmask(xs0), ys0)
                    yerrs0 = np.ma.masked_where(np.ma.getmask(xs0), yerrs0)
                if np.ma.count_masked(xs0) < np.ma.count_masked(ys0):
                    xs0 = np.ma.masked_where(np.ma.getmask(ys0), xs0)
                    xerrs0 = np.ma.masked_where(np.ma.getmask(ys0), xerrs0)
                xs = xs0.compressed() * scale[0]
                ys = ys0.compressed() * scale[1]
                xerrs = xerrs0.compressed() * scale[0]
                yerrs = yerrs0.compressed() * scale[1]
                if ax1ax2[k] == 0:
                    if (lloc == 0 and j == 0 and k == [0 if round((data.shape[2]) / 2) < 2 else 1][0]):
                        if bars is True:
                            ax.errorbar(xs, ys, color=colors[c1], linestyle=linestyles[d], marker=markers[0],
                                        xerr=xerrs, yerr=yerrs, capsize=5, label=labels[l])
                        else:
                            ax.errorbar(xs, ys, color=colors[c1], linestyle=linestyles[d], marker=markers[0], capsize=5,
                                        label=labels[l])

                        l += 1
                    elif (lloc == 1 and d == 0 and k == [0 if round((data.shape[2]) / 2) < 2 else 1][0]):
                        if bars is True:
                            ax.errorbar(xs, ys, color=colors[c1], linestyle=linestyles[d], marker=markers[0],
                                        xerr=xerrs, yerr=yerrs, capsize=5, label=labels[l])
                        else:
                            ax.errorbar(xs, ys, color=colors[c1], linestyle=linestyles[d], marker=markers[0], capsize=5,
                                        label=labels[l])
                        l += 1
                    elif (lloc == 2 and d == 0 and j == 0):
                        if bars is True:
                            ax.errorbar(xs, ys, color=colors[c1], linestyle=linestyles[d], marker=markers[0],
                                        xerr=xerrs, yerr=yerrs, capsize=5, label=labels[l])
                        else:
                            ax.errorbar(xs, ys, color=colors[c1], linestyle=linestyles[d], marker=markers[0], capsize=5,
                                        label=labels[l])
                        l += 1
                    else:
                        if bars is True:
                            ax.errorbar(xs, ys, color=colors[c1], linestyle=linestyles[d], marker=markers[0],
                                        xerr=xerrs, yerr=yerrs, capsize=5)
                        else:
                            ax.errorbar(xs, ys, color=colors[c1], linestyle=linestyles[d], marker=markers[0], capsize=5)
                    c1 += 1

                else:
                    if (lloc == 0 and j == 0 and k == [0 if round((data.shape[2]) / 2) < 2 else 1][0]):
                        if bars is True:
                            # ax2.errorbar(xs, ys, fmt=form, xerr=xerrs, yerr=yerrs, capsize=5, label=labels[l])
                            ax2.errorbar(xs, ys, color=colors[c2], linestyle=linestyles[d], marker=markers[1],
                                         xerr=xerrs, yerr=yerrs, capsize=5, label=labels[l])

                        else:
                            # ax2.errorbar(xs, ys, fmt=form, capsize=5, label=labels[l])
                            ax2.errorbar(xs, ys, color=colors[c2], linestyle=linestyles[d], marker=markers[1],
                                         capsize=5, label=labels[l])

                        l += 1
                    elif (lloc == 1 and d == 0 and k == [0 if round((data.shape[2]) / 2) < 2 else 1][0]):
                        if bars is True:
                            ax2.errorbar(xs, ys, color=colors[c2], linestyle=linestyles[d], marker=markers[1],
                                         xerr=xerrs, yerr=yerrs, capsize=5, label=labels[l])
                        else:
                            ax2.errorbar(xs, ys, color=colors[c2], linestyle=linestyles[d], marker=markers[1],
                                         capsize=5, label=labels[l])
                        l += 1

                    elif (lloc == 2 and d == 0 and j == 0):
                        if bars is True:
                            ax2.errorbar(xs, ys, color=colors[c2], linestyle=linestyles[d], marker=markers[1],
                                         xerr=xerrs, yerr=yerrs, capsize=5, label=labels[l])
                        else:
                            ax2.errorbar(xs, ys, color=colors[c2], linestyle=linestyles[d], marker=markers[1],
                                         capsize=5, label=labels[l])
                        l += 1
                    else:
                        if bars is True:
                            ax2.errorbar(xs, ys, color=colors[c2], linestyle=linestyles[d], marker=markers[1],
                                         xerr=xerrs, yerr=yerrs, capsize=5)
                        else:
                            ax2.errorbar(xs, ys, color=colors[c2], linestyle=linestyles[d], marker=markers[1],
                                         capsize=5)
                    c2 += 1
        d += 1
    if power_law[0] != 0:
        xbounds = ax.get_xbound()
        x = np.linspace(xbounds[0] + 0.1 * (xbounds[1] - xbounds[0]), xbounds[0] + 0.66 * (xbounds[1] - xbounds[0]),50)
        y = power_law[0] * x ** power_law[1]
        ax.plot(x, y, 'k--', label=r'$%s \sim %s^{%.2f}$' % (power_law[2], power_law[3], power_law[1]))
    if log[0] is True:
        ax.set_xscale('log')
    if log[1] is True:
        ax.set_yscale('log')
        if axis2 != None:
            ax2.set_yscale('log')

    if (log[0] is True) or (log[1] is True):
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    else:
        ax.set(xlim=xlim, xlabel=xlabel, ylabel=ylabel, title=title)

    if axis2 != None:
        ax.legend(loc=8)
        ax2.legend(loc=4)
        ax2.minorticks_on()
        ax2.set(ylabel=y2label)
    else:
        ax.legend(loc=0)
    ax.minorticks_on()
    ax.grid(visible=True, which='both', axis='both')
    axs.append(ax)
    if axis2 != None:
        axs.append(ax2)
    return axs

def myFigs(datas,styles=['-','--','-.',':'],llabels=(None,None),xlabel=None,ylabel=None,title=None,xlim=False,axis2=None,y2label=None,log=(False,False)):
    lloc = llabels[0]
    labels = llabels[1]
    for d in datas:
        while len(d.shape) < 4:
            datas[d] = np.expand_dims(datas[d], axis=0)
    figs = []
    for i in range(datas[0].shape[0]):
        fig, ax = plt.subplots()
        if axis2 != None:
            ax2 = ax.twinx()
        l = 0
        d = 0
        for data in datas:
            if axis2 is None:
                ax1ax2 = [0] * round((data.shape[2])/2)
            else:
                ax1ax2 = axis2.copy()
            nlines = data.shape[1] * (data.shape[2] / 2)
            if nlines > 4:
                colors = ['r', 'y', 'g', 'c', 'b', 'm', 'k'] * 5
            elif nlines == 1:
                colors = ['k']
            else:
                colors = ['r', 'g', 'b', 'k'] * 10
            c = 0
            for j in range(data.shape[1]):
                for k in range(round((data.shape[2])/2)):
                    form = '{0}{1}'.format(styles[d],colors[c])
                    if ax1ax2[k]==0:
                        if(lloc==0 and j==0 and k==[0 if round((data.shape[2])/2)<2 else 1][0] ):
                            ax.plot(data[i, j, k * 2, :].compressed(), data[i, j, (k * 2) + 1, :].compressed(), form,label=labels[l] )
                            l+=1
                        elif(lloc==1 and d==0 and k==[0 if round((data.shape[2])/2)<2 else 1][0]):
                            ax.plot(data[i, j, k * 2, :].compressed(), data[i, j, (k * 2) + 1, :].compressed(), form,label=labels[l] )
                            l+=1
                        elif(lloc==2 and d==0 and j==0):
                            ax.plot(data[i, j, k * 2, :].compressed(), data[i, j, (k * 2) + 1, :].compressed(), form,label=labels[l] )
                            l+=1
                        else:
                            ax.plot(data[i, j, k * 2, :].compressed(), data[i, j, (k * 2) + 1, :].compressed(), form)
                    else:
                        if (lloc == 0 and j == 0 and k == [0 if round((data.shape[2]) / 2) < 2 else 1][0]):
                            ax2.plot(data[i, j, k * 2, :].compressed(), data[i, j, (k * 2) + 1, :].compressed(), form,
                                    label=labels[l])
                            l += 1
                        elif (lloc == 1 and d == 0 and k==[0 if round((data.shape[2])/2)<2 else 1][0]):
                            ax2.plot(data[i, j, k * 2, :].compressed(), data[i, j, (k * 2) + 1, :].compressed(), form,
                                    label=labels[l])
                            l += 1
                        elif (lloc == 2 and d == 0 and j == 0):
                            ax2.plot(data[i, j, k * 2, :].compressed(), data[i, j, (k * 2) + 1, :].compressed(), form,
                                    label=labels[l])
                            l += 1
                        else:
                            ax2.plot(data[i, j, k * 2, :].compressed(), data[i, j, (k * 2) + 1, :].compressed(), form)

                    c+=1
            d+=1
        ax.set(xlim=xlim, xlabel=xlabel, ylabel=ylabel, title=title)
        if log[0] is True:
            plt.xscale('log')
        if log[1] is True:
            plt.yscale('log')
        if axis2 != None:
            ax.legend(loc=8)
            ax2.legend(loc=4)
            ax2.minorticks_on()
            ax2.set(ylabel=y2label)
        else:
            ax.legend(loc=0)
        ax.minorticks_on()
        ax.grid(visible=True, which='both', axis='both')
        figs.append(fig)
        return figs


def myCOF(PvS):
    dims = (PvS.shape[0],PvS.shape[1],PvS.shape[2],2,2)
    FvS = np.zeros(dims)
    for i in range(PvS.shape[0]):
        for j in range(PvS.shape[1]):
            for k  in range(PvS.shape[2]):
                FvS[i,j,k,0,0] = PvS[i,j,k,0,0]
                FvS[i,j,k,1,0] = -(PvS[i,j,k,1,0])/(PvS[i,j,k,3,0])
                FvS[i,j,k,0,1] = PvS[i, j, k, 0, 1]
                FvS[i,j,k,1,1] = abs(FvS[i,j,k,1,0]) * math.sqrt( (PvS[i,j,k,1,1]/PvS[i,j,k,1,0])**2 + (PvS[i,j,k,3,1]/PvS[i,j,k,3,0])**2 )
    B = np.ma.masked_invalid(FvS)
    return B

def myBrushHeight(zs,dps,f=0.95):
    total = np.trapz(dps,zs)
    i = 10
    cumulative = 0
    if total == 0.0:
        h = np.nan
    else:
        while(cumulative/total <= f):
            cumulative = np.trapz(dps[0:i], zs[0:i])
            i += 1
        h = zs[i]
    return h

def myCrit_Srate(srate,R2x):
    threshold = 1.3
    flag = 0
    Vi = 0
    for i in range(srate.shape[0]-1):
        if (R2x[i] > threshold and flag == 0 and R2x[i+1]>R2x[i]):
            Vi = i-1
            flag = 1
    if flag == 0:
        Vi = 0
    return srate[Vi]

def mainVar(fname,var):
    fLine = myFindLine(fname,['variable ', ' {0} '.format(var)])[0]
    val = pd.read_table(fname,sep="\s*[,]\s*",skiprows = fLine,nrows=1 , header=None, index_col=None,comment='#',engine='python').values[0][0].split()[3:]
    return val[0]

def simVar(log,var):
    # print(log)
    fLine = myFindLine(log,['Variable[', ' {0} '.format(var)])[0]
    val = pd.read_table(log,skiprows = fLine,nrows=1 , header=None, index_col=None,comment='#',engine='python').values[0][0].split()[-2]
    return float(val)

def Rg2x(RvS):
    dims = RvS.shape
    R2vS = np.zeros(dims)
    R0x_vs = RvS[:, 1, 0]
    R0x_i = np.where(R0x_vs == np.nanmin(R0x_vs))[0][0]
    R0x = ufloat(RvS[R0x_i, 1, 0], RvS[R0x_i, 1, 1])
    for i in range(RvS.shape[0]):
        gamma_dot = ufloat(RvS[i, 0, 0], RvS[i,0, 1])
        Rgx = ufloat(RvS[i, 1, 0], RvS[i,1, 1])
        Rg2x_Rg20x = (Rgx**2 / R0x**2)
        R2vS[i, 0, 0] = gamma_dot.nominal_value
        R2vS[i, 0, 1] = gamma_dot.std_dev
        R2vS[i, 1, 0] = Rg2x_Rg20x.nominal_value
        R2vS[i, 1, 1] = Rg2x_Rg20x.std_dev
    B = np.ma.masked_invalid(R2vS)
    return B

def srate_crit(R2xvS):
    srate = myCrit_Srate(R2xvS[:,0,0],R2xvS[:,1,0])
    B = np.ma.masked_invalid(srate)
    return B

def Weissenberg(R2xvS,src):
    R2xvW = np.copy(R2xvS)
    for i in range(R2xvS.shape[0]):
        R2xvW[:,0,0] = ((R2xvS[:,0,0]) / src)
    B = np.ma.masked_invalid(R2xvW)
    return B

def COF(PvS):
    dims = (PvS.shape[0],2,2)
    FvS = np.zeros(dims)
    for i in range(PvS.shape[0]):
        gamma_dot = ufloat(PvS[i, 0, 0], PvS[i, 0, 1])
        pxz = ufloat(PvS[i, 1, 0], PvS[i, 1, 1])
        pzz = ufloat(PvS[i, 3, 0], PvS[i, 3, 1])
        cof = - pxz / pzz
        FvS[i, 0, 0] = gamma_dot.nominal_value
        FvS[i, 0, 1] = gamma_dot.std_dev
        FvS[i, 1, 0] = cof.nominal_value
        FvS[i, 1, 1] = cof.std_dev
    B = np.ma.masked_invalid(FvS)
    return B

class Log():
    def __init__(self, path):
        self.path = path
        self.start_tag = 'Step Temp Press'
        self.end_tag = 'Loop time of'
        self.rst_tag = 'print "<<<'
        self.starts = []
        self.start_timesteps = []
        self.ends0 = []
        with open(self.path, 'r') as f:
            started = False
            found_start = False
            prev_line = 0
            for n, line in enumerate(f):
                if self.start_tag in line:
                    started = True
                elif self.end_tag in line and started:
                    started = False
                    found_start = False
                    self.ends0.append(n-2)
                elif self.rst_tag in line and started:
                    started = False
                    found_start = False
                    self.ends0.append(n-2)
                else:
                    pass
                prev_line = n
                if started and not found_start and line[0:8].strip().isnumeric():
                    found_start = True
                    self.starts.append(n)
                    self.start_timesteps.append(int(line.split()[0]))
            if started:
                self.ends0.append(prev_line-2)
        # print(self.path)
        # print(self.starts)
        twosteps = np.loadtxt(self.path, dtype=int, comments=['#', 'W'], skiprows=self.starts[0], usecols=(0),max_rows=3)
        self.thermo_step = twosteps[1] - twosteps[0]
        entries0 = [end - (start-1) for start,end in list(zip(self.starts,self.ends0))]
        # This part is to remove any zero entry blocks and let only the final block go till the full end
        i = 0
        for entry in entries0:
            if entry == 0:
                self.starts.pop(i)
                self.ends0.pop(i)
                entries0.pop(i)
                self.start_timesteps.pop(i)
            i += 1
        self.ends0[-1] = self.ends0[-1] + 1
        entries0[-1] = entries0[-1] + 1
        self.file_end_timesteps = [start_timestep + ((entries0-1) * self.thermo_step) for start_timestep,entries0 in list(zip(self.start_timesteps,entries0))]
        self.ends = []
        i = 0
        overshoots = []
        for entries in entries0:
            if entries is not entries0[-1]:
                if (self.file_end_timesteps[i] > self.start_timesteps[i+1]):
                    overshoot = int((self.file_end_timesteps[i] - self.start_timesteps[i+1])/self.thermo_step) + 1
                    overshoots.append(overshoot)
                else:
                    overshoots.append(0)
            else:
                overshoots.append(0)
            i += 1
        self.ends = [ends0 - overshoot for ends0,overshoot in list(zip(self.ends0,overshoots)) ]
        self.entries = [entries0 - overshoot for entries0,overshoot in list(zip(entries0,overshoots)) ]
        # print('Starts:', self.starts)
        # print('Ends:',self.ends)
        # print('Entries:',self.entries)

    def get_data(self,cols, steps = 40, avg=True):
        # print("1",self.path)
        # print(steps)
        total_entries = sum([entries for entries in self.entries if entries>1])
        if (steps > total_entries) or (steps == 0):
            steps = total_entries
        steps_toread = steps
        # print(0,steps_toread)
        i = 1
        datas = []
        # print(self.entries)
        while steps_toread > 1:
            if (self.entries[- i] > 1):
                if steps_toread <= self.entries[- i]:
                    start_shift = self.entries[- i] - steps_toread
                    new_start = ((self.starts[-i]) + start_shift)
                    data0 = np.loadtxt(self.path, skiprows = new_start , usecols=cols, max_rows=steps_toread)
                    steps_toread = 0
                    datas.append(data0)
                elif steps_toread > self.entries[- i]:
                    steps_toread = steps_toread - self.entries[- i]
                    data0 = np.loadtxt(self.path, skiprows = self.starts[-i] , usecols=cols, max_rows=self.entries[- i])
                    datas.append(data0)
            # print(2,steps_toread)
            i += 1
        # print(datas)
        datas.reverse()
        data = np.concatenate(datas,axis = 0)
        if avg:
            means = np.mean(data, axis=0)
            errors = np.std(data, axis=0)
            data = np.vstack((means, errors)).T
        return data

class Run():

    def __init__(self,path):
        self.path = path
        self.M = int(mainVar(self.path + '\main.in',' M '))
        self.N = int(mainVar(self.path + '\main.in',' N '))
        self.xhi = int(mainVar(self.path + '\main.in',' xhi '))
        self.yhi = int(mainVar(self.path + '\main.in',' yhi '))
        self.N_s = int(mainVar(self.path + '\main.in',' N_s '))
        self.M_s = int(mainVar(self.path + '\main.in',' M_s '))
        self.N_m = (self.N - (self.M_s * self.N_s))
        self.R_N_scale = float(mainVar(self.path + '\main.in',' R_N_scale'))
        self.R_M_scale = float(mainVar(self.path + '\main.in',' R_M_scale'))
        self.R_frac = float(mainVar(self.path + '\main.in',' R_frac'))
        self.L_N_scale = float(mainVar(self.path + '\main.in',' L_N_scale'))
        self.L_M_scale = float(mainVar(self.path + '\main.in',' L_M_scale'))
        self.L_frac = float(mainVar(self.path + '\main.in',' L_frac'))
        self.M_Ri = (math.floor((self.R_frac * self.R_M_scale) * (self.M)))
        self.M_Lo = (math.floor((self.L_frac * self.L_M_scale) * (self.M)))
        self.M_ln = (self.M - (self.M_Ri/self.R_M_scale) - (self.M_Lo/self.L_M_scale))
        self.M_chains = (self.M_ln + self.M_Ri + self.M_Lo)
        self.G = (self.M_ln + self.M_Ri + (self.M_Lo/self.L_M_scale))
        self.GD = self.G/(self.xhi * self.yhi)
        self.Dcomp = int(mainVar(self.path + '\main.in',' Dcomp '))
        Vdirs0 = list(filter(lambda x: 'V=' in x,os.listdir(self.path)))
        Vs0 = [float(x.split('V=')[1]) for x in Vdirs0]
        self.Vdirs = [x for _, x in sorted(zip(Vs0, Vdirs0))]
        self.Vs = sorted(Vs0)
        if len(self.Vdirs) == 0:
            self.Vdirs = self.mainVar('Vwalli').copy()
        thermoline = myFindLine(self.path + '\ecs.in',['thermo_style',' custom ',' step ',' etotal ', ' ke '])
        self.tkeys = myReadLines(self.path + '\ecs.in',[thermoline])[0].split()[2:]
        self.tdict = {self.tkeys[i] : i+1  for i in range(len(self.tkeys))}

        return

    def mainVar(self, var):
        fname = self.path + r'\main.in'
        val = mainVar(fname, var)
        return val[0]

    def thermo(self,stage, tvars, steps = 40, Vis = [0], avg = True):
        cols = [self.tdict[tvar] for tvar in tvars]
        mycols = cols.copy()
        pdcols = sorted(mycols)
        tvar_order = [pdcols.index(i) for i in mycols]
        cols.insert(0,0)
        if Vis == 0 or len(self.Vdirs)==0:
            Varray = self.Vdirs.copy()
        else:
            Varray = []
            for Vi in Vis:
                Varray.append(self.Vdirs[Vi])
        if stage == 1:
            datafile = 'equil.csv'
        elif stage == 2:
            datafile = 'comp.csv'
        else:
            datafile = 'shear.csv'
        if len(tvars) == 1:
            ax = 1
        else:
            ax  = (len(tvars)-1)*2

        B = []
        for Vstr in Varray:
            fname = r'{0}\{1}\{2}'.format(self.path,Vstr,datafile)
            filesize = os.path.getsize(fname)
            if (os.path.isfile(fname) == False):
                if avg == False:
                    val = np.ones((ax,steps)) * np.nan
                else:
                    val = np.ones((ax,2)) * np.nan
            else:
                entries = pd.read_csv(fname,header=0, usecols=[0],engine='python').shape[0]
                skip_rows =  entries - steps
                skip_foot = 0 # entries - end_row
                df = pd.read_csv(fname,header=0,index_col=0,usecols=cols,skiprows=skip_rows,skipfooter=skip_foot,engine='python')
                val0 = df.values
                val1 = np.array([val0[:, i] for i in tvar_order])
                val = np.array([val1[(math.ceil(i/2)),:] if (i%2 != 0) else val1[0,:] for i in range(ax)])
                if (avg==True):
                    means = df.mean(axis=0).values
                    errors = df.sem(axis=0).values
                    val0 = np.array([means,errors]).T
                    val1 = np.array([val0[i] for i in tvar_order])
                    val = np.array([val1[(math.ceil(i/2)),:] if (i%2 != 0) else val1[0,:] for i in range(ax)])
            B.append(val)
        C = np.asarray(B)
        E = np.ma.masked_invalid(C)
        return E

    def profiles(self,datafiles,Vis = [0],bins=1000, Z2D = True ):

        if Vis == 0 or len(self.Vdirs)==0:
            Varray = self.Vdirs.copy()
        else:
            Varray = []
            for Vi in Vis:
                Varray.append(self.Vdirs[Vi])
        B = []
        for Vstr in Varray:
            # C = []
            C = np.zeros((2*len(datafiles),bins))
            i = 0
            for datafile in datafiles:
                fname = r'{0}\{1}\{2}.csv'.format(self.path, Vstr, datafile)
                if (os.path.isfile(fname) == False) or filesize < 200:
                    C[i*2,:] = np.ones((1,bins)) * np.nan
                    C[(i*2)+1,:] = np.ones((1,bins)) * np.nan
                else:
                    tcols = len(pd.read_csv(fname,header=0,index_col=0,nrows=1).values[0])
                    df = pd.read_csv(fname, header=0, index_col=0,usecols=[0,1,tcols])
                    val0 = df.values.T
                    if 'pe' in datafile:
                        stage = 1
                    elif 'pc' in datafile:
                        stage = 2
                    else:
                        stage = 3
                    brush = Run(self.path)
                    ans = brush.thermo(stage, ['v_blzmax', 'v_D_m'],Vis=[self.Vdirs.index(Vstr)], avg=True)
                    blzmax = ans[0, 0, 0]
                    D = ans[0, 1, 0]
                    if Z2D == True:
                        val0[0,:] = (val0[0,:] - blzmax )/D
                    else:
                        val0[0,:] = (val0[0,:] - blzmax)
                    val = val0
                    C[(i*2),:] = val[0,-bins:]
                    C[(i*2)+1,:] = val[1,-bins:]
                    i += 1
            B.append(C)
        E = np.asarray(B)
        F = np.ma.masked_invalid(E)
        return F

class Bilayer():

    def __init__(self,path):
        self.path = path
        self.N = int(path[(path.find('N=')+2):path.find(r'\M=')])
        self.M = int(path[(path.find('M='))+2:path.find(r'\X=')])
        self.X = int(path[(path.find('X=')+2):])
        mainpath = self.path + r'\PBB-ECS\main.in'
        self.Wall_control = ' '
        try:
            self.Wall_control_l = myFindLine(mainpath, ['variable ', ' Wall_control '])[0]
            self.Wall_control = myReadLines(mainpath, [self.Wall_control_l])[0].split('variable       Wall_control string ')[1][0]
        except:
            pass
        PD = ['D=' if 'D' in self.Wall_control else 'P='][0]
        PDdirs0 = os.listdir(self.path)
        # print(self.path)
        PDdirs0.remove('PBB-ECS')
        PDs0 = [float(x.split(PD)[1]) for x in PDdirs0]
        self.PDdirs = [x for _,x in sorted(zip(PDs0,PDdirs0))]
        self.PDs = sorted(PDs0)
        v_n = []
        for pd in range(len(self.PDdirs)):
            Vdirs1 = list(filter(lambda x : 'V=' in x, os.listdir(r'{0}\{1}\PBB-ECS'.format(self.path,self.PDdirs[pd]))))
            v_n.append(len(Vdirs1))
        Vdirs0 = list(filter(lambda x : 'V=' in x, os.listdir(r'{0}\{1}\PBB-ECS'.format(self.path,self.PDdirs[v_n.index(max(v_n))]))))
        Vs0 = [float(x.split('V=')[1]) for x in Vdirs0]
        self.VdirsX = [x for _,x in sorted(zip(Vs0,Vdirs0))]
        self.Vs = sorted(Vs0)
        self.Vs_main_str = sorted(self.mainVar('Vwalli').copy())
        self.Vs_main = [float(x) for x in self.Vs_main_str]
        lm = len (self.Vs_main)
        # if len(self.Vdirs) == 0:
        #     self.Vdirs = self.Vs_main
        self.Vmiss = [x if x in self.Vs else "missing" for x in self.Vs_main]
        j = 0
        self.Vdirs = []
        for V in self.Vmiss:
            if V == "missing":
                self.Vdirs.append('missing')
            else:
                self.Vdirs.append(self.VdirsX[j])
                j += 1
        # print('R',self.Vdirs)

        ld = len(self.Vs)
        ecspath = r'{0}\PBB-ECS\ecs.in'.format(self.path)
        thermoline = myFindLine(ecspath,['thermo_style',' custom ',' step ',' f_pe ', ' f_temp '])[0]
        self.tkeys = myReadLines(ecspath,[thermoline])[0].split()[2:]
        self.tdict = {self.tkeys[i] : i+1  for i in range(len(self.tkeys))}
        return

    def mainVar(self,var):
        fname = self.path + r'\PBB-ECS\main.in'
        fLine = myFindLine(fname,['variable ', ' {0} '.format(var)])[0]
        val = pd.read_table(fname,sep="\s*[,]\s*",skiprows = fLine,nrows=1 , header=None, index_col=None,comment='#',engine='python').values[0][0].split()[3:]
        return val

    def mainVars(self,mvar, PDis = [0],Vis = [0]):
        if PDis == 0:
            PDarray = self.PDdirs.copy()
        else:
            PDarray = []
            for PDi in PDis:
                PDarray.append(self.PDdirs[PDi])
        if Vis == 0 or len(self.Vdirs) == 0:
            Varray = self.Vdirs.copy()
        else:
            Varray = []
            for Vi in Vis:
                try:
                    Varray.append(self.Vdirs[Vi])
                except:
                    Varray.append('missing-dir')
        ax = 1
        A = []
        for PDstr in PDarray:
            B = []
            for Vstr in Varray:
                main = os.path.join(self.path, PDstr, 'PBB-ECS', 'main.in')
                if os.path.isfile(main):
                    # print(main)
                    val = np.zeros((ax, 2))
                    var = float(mainVar(main, mvar))
                    val[0, 0] = var
                else:
                    print("missing main file")
                B.append(val)
            A.append(B)
        C = np.asarray(A)
        return C


    def simVars0(self, svar, PDis = [0],Vis = [0]):
        if PDis == 0:
            PDarray = self.PDdirs.copy()
        else:
            PDarray = []
            for PDi in PDis:
                PDarray.append(self.PDdirs[PDi])
        if Vis == 0 or len(self.Vdirs) == 0:
            Varray = self.Vdirs.copy()
        else:
            Varray = []
            for Vi in Vis:
                try:
                    Varray.append(self.Vdirs[Vi])
                except:
                    Varray.append('miss')
        ax = 1
        A = []
        for PDstr in PDarray:
            B = []
            log = os.path.join(self.path, PDstr, 'PBB-ECS', 'comp.log')
            if os.path.isfile(log):
                var = float(simVar(log, svar))
            else:
                var = np.nan
            for Vstr in Varray:
                fname = os.path.join(self.path, PDstr, 'PBB-ECS', Vstr, 'shear.log')
                # print(fname)
                if os.path.isfile(fname):
                    filesize = os.path.getsize(fname)
                    if filesize < 200:
                        val = np.ones((ax, 2)) * np.nan
                    else:
                        val = np.zeros((ax, 2))
                        val[0, 0] = var
                else:
                    val = np.ones((ax, 2)) * np.nan
                B.append(val)
            A.append(B)
        C = np.asarray(A)
        return C

    def simVars(self, svar, PDis = [0],Vis = [0]):
        if PDis == 0:
            PDarray = self.PDdirs.copy()
        else:
            PDarray = []
            for PDi in PDis:
                PDarray.append(self.PDdirs[PDi])
        if Vis == 0 or len(self.Vdirs) == 0:
            Varray = self.Vdirs.copy()
        else:
            Varray = []
            for Vi in Vis:
                try:
                    Varray.append(self.Vdirs[Vi])
                except:
                    Varray.append('miss')
        ax = 1
        A = []
        for PDstr in PDarray:
            B = []
            for Vstr in Varray:
                # if 'V=0.02999999999999999889' in Vstr:
                #     log = os.path.join(self.path, PDstr, 'PBB-ECS', 'comp.log')
                # else:
                #     log = os.path.join(self.path, PDstr, 'PBB-ECS', Vstr, 'log.lammps')
                log = os.path.join(self.path, PDstr, 'PBB-ECS', Vstr, 'log.lammps')
                log0 = os.path.join(self.path, PDstr, 'PBB-ECS', 'comp.log')
                fname = os.path.join(self.path, PDstr, 'PBB-ECS', Vstr, 'shear.log')
                # print(fname)
                if os.path.isfile(log) and os.path.isfile(fname):
                    filesize = os.path.getsize(fname)
                    if filesize < 200:
                        val = np.ones((ax, 2)) * np.nan
                    else:
                        try:
                            var = float(simVar(log, svar))
                        except:
                            var = float(simVar(log0, svar))
                        val = np.zeros((ax, 2))
                        val[0, 0] = var
                else:
                    val = np.ones((ax, 2)) * np.nan
                B.append(val)
            A.append(B)
        C = np.asarray(A)
        return C


    def type(self):
        if self.mainVar('L') == 0:
            if self.mainVar('Ct') == 0:
                type = 'Linear'
            elif self.mainVar('Ct') == 1:
                type = 'Cyclic'
            else:
                type = str(self.mainVar)+'Linear-Cyclic'
        else:
            type = 'Looped'

        return type

    def gd(self):
        xhi = self.mainVar('xhi')
        yhi = self.mainVar('yhi')
        val = self.M/(xhi * yhi)
        return val

    def thermo0(self,stage, tvars, steps = 40, PDis = [0],Vis = [0], avg = True):
        cols = [self.tdict[tvar] for tvar in tvars]
        mycols = cols.copy()
        pdcols = sorted(mycols)
        tvar_order = [pdcols.index(i) for i in mycols]
        cols.insert(0,0)
        if PDis == 0:
            PDarray = self.PDdirs.copy()
        else:
            PDarray = []
            for PDi in PDis:
                PDarray.append(self.PDdirs[PDi])
        if Vis == 0 or len(self.Vdirs) == 0:
            Varray = self.Vdirs.copy()
        else:
            Varray = []
            for Vi in Vis:
                try:
                    Varray.append(self.Vdirs[Vi])
                except:
                    Varray.append('missing-dir')
        if stage == 1:
            datafile = 'equil.log'
        elif stage == 2:
            datafile = 'precomp.log'
        else:
            datafile = 'shear.log'
        if len(tvars) == 1:
            ax = 1
        else:
            ax  = (len(tvars)-1)*2
        A = []
        for PDstr in PDarray:
            B = []
            for Vstr in Varray:
                fname = r'{0}\{1}\PBB-ECS\{2}\{3}'.format(self.path,PDstr,Vstr,datafile)

                if os.path.isfile(fname):
                    filesize = os.path.getsize(fname)
                    if filesize < 200:
                        if avg == False:
                            val = np.ones((ax, steps)) * np.nan
                        else:
                            val = np.ones((ax, 2)) * np.nan
                    else:
                        entries = pd.read_csv(fname, header=0, usecols=[0], engine='python').shape[0]
                        skip_rows = entries - steps
                        skip_foot = 0  # entries - end_row
                        df = pd.read_csv(fname, header=0, index_col=0, usecols=cols, skiprows=skip_rows,
                                         skipfooter=skip_foot, engine='python')
                        val0 = df.values
                        val1 = np.array([val0[:, i] for i in tvar_order])
                        val = np.array([val1[(math.ceil(i / 2)), :] if (i % 2 != 0) else val1[0, :] for i in range(ax)])
                        if (avg == True):
                            means = df.mean(axis=0).values
                            errors = df.sem(axis=0).values
                            val0 = np.array([means, errors]).T
                            val1 = np.array([val0[i] for i in tvar_order])
                            val = np.array([val1[(math.ceil(i / 2)), :] if (i % 2 != 0) else val1[0, :] for i in range(ax)])
                else:
                    if avg == False:
                        val = np.ones((ax, steps)) * np.nan
                    else:
                        val = np.ones((ax, 2)) * np.nan

                B.append(val)
            A.append(B)
        C = np.asarray(A)
        return C

    def thermo(self,stage, tvars,steps = 40, PDis = [0],Vis = [0], avg = True):
        # print(self.tdict)
        cols = [self.tdict[tvar] - 1 for tvar in tvars]
        cols=tuple(cols)
        if PDis == 0:
            PDarray = self.PDdirs.copy()
        else:
            PDarray = []
            for PDi in PDis:
                PDarray.append(self.PDdirs[PDi])
        if Vis == 0:
            Varray = self.Vdirs.copy()
        else:
            Varray = []
            for Vi in Vis:
                try:
                    Varray.append(self.Vdirs[Vi])
                except:
                    Varray.append('missing-dir')
        if stage == 1:
            datafile = 'equil.log'
        elif stage == 2:
            datafile = 'comp.log'
        else:
            datafile = 'shear.log'
        if len(tvars) == 1:
            ax = 1
        else:
            ax = (len(tvars)-1)*2
        A = []
        for PDstr in PDarray:
            B = []
            for Vstr in Varray:
                # fname = r'{0}\{1}\PBB-ECS\{2}\{3}'.format(self.path,PDstr,Vstr,datafile)
                if datafile == 'shear.log':
                    fname = os.path.join(self.path, PDstr, 'PBB-ECS', Vstr, datafile)
                else:
                    fname = os.path.join(self.path, PDstr, 'PBB-ECS', datafile)
                if os.path.isfile(fname):
                    # print(fname)
                    filesize = os.path.getsize(fname)
                    last_line = myLastLine(fname)
                    CompletedVs = myCompletedVs(os.path.join(self.path, PDstr, 'PBB-ECS'))
                    if filesize < 200 or ('Last command' in last_line) or len(CompletedVs) < 1:
                        if avg == False:
                            val = np.ones((ax, steps)) * np.nan
                        else:
                            val = np.ones((ax, 2)) * np.nan
                    else:
                        log = Log(fname)
                        data = log.get_data(cols=cols,steps=steps)
                        if len(data.shape) == 1:
                            data = np.expand_dims(data,axis=-1)

                        val1 = data
                        val = np.array([val1[(math.ceil(i / 2)), :] if (i % 2 != 0) else val1[0, :] for i in range(ax)])
                else:
                    if avg == False:
                        val = np.ones((ax, steps)) * np.nan
                    else:
                        val = np.ones((ax, 2)) * np.nan

                B.append(val)
            A.append(B)
        C = np.asarray(A)
        return C
    def thermoNM(self,stage, PDis = [0],Vis = [0]):
        if PDis == 0:
            PDarray = self.PDdirs.copy()
        else:
            PDarray = []
            for PDi in PDis:
                PDarray.append(self.PDdirs[PDi])
        if Vis == 0 or len(self.Vdirs) == 0:
            Varray = self.Vdirs.copy()
        else:
            Varray = []
            for Vi in Vis:
                try:
                    Varray.append(self.Vdirs[Vi])
                except:
                    Varray.append('missing-dir')
        if stage == 1:
            datafile = 'equil.log'
        elif stage == 2:
            datafile = 'comp.log'
        else:
            datafile = 'shear.log'
        ax = 2
        A = []
        for PDstr in PDarray:
            B = []
            main = os.path.join(self.path, PDstr,'PBB-ECS','main.in')
            for Vstr in Varray:
                # fname = r'{0}\{1}\PBB-ECS\{2}\{3}'.format(self.path,PDstr,Vstr,datafile)
                if datafile == 'shear.log':
                    fname = os.path.join(self.path, PDstr, 'PBB-ECS', Vstr, datafile)
                else:
                    fname = os.path.join(self.path, PDstr, 'PBB-ECS', datafile)
                if os.path.isfile(fname):
                    filesize = os.path.getsize(fname)
                    if filesize < 200:
                        val = np.ones((ax, 2)) * np.nan
                    else:
                        val = np.zeros((ax, 2))
                        N = int(mainVar(main, 'N'))
                        M = int(mainVar(main, 'M'))
                        val[0, 0] = N
                        val[1, 0] = M
                else:
                    val = np.ones((ax, 2)) * np.nan

                B.append(val)
            A.append(B)
        C = np.asarray(A)
        return C

    def profiles(self,datafiles,PDis = [0],Vis = [0],bins=1000, Z2D = True ):
        if PDis == 0:
            PDarray = self.PDdirs.copy()
        else:
            PDarray = []
            for PDi in PDis:
                PDarray.append(self.PDdirs[PDi])

        if Vis == 0 or len(self.Vdirs)==0:
            Varray = self.Vdirs.copy()
        else:
            Varray = []
            for Vi in Vis:
                try:
                    Varray.append(self.Vdirs[Vi])
                except:
                    Varray.append('missing-dir')
        A = []
        for PDstr in PDarray:
            B = []
            for Vstr in Varray:
                C = np.zeros((2*len(datafiles),bins))
                i = 0
                for datafile in datafiles:
                    PBBname = os.path.join(self.path,PDstr,'PBB-ECS')
                    CompletedVs = [os.path.join(PBBname, o) for o in os.listdir(PBBname) if os.path.isdir(os.path.join(PBBname,o))]
                    if datafile in ['abeads_sdz','bbeads_sdz','tbeads_sdz','sbeads_sdz','temp_sz','velp_sz']:
                        fname = r'{0}\{1}\PBB-ECS\{2}\{3}'.format(self.path, PDstr, Vstr, datafile)
                    else:
                        fname = r'{0}\{1}\PBB-ECS\{2}'.format(self.path, PDstr, datafile)
                    # print(fname)
                    if os.path.isfile(fname):
                        filesize = os.path.getsize(fname)
                        CompletedVs = myCompletedVs(os.path.join(self.path, PDstr, 'PBB-ECS'))
                        if filesize < 200 or (len(CompletedVs) < 1):
                            val = np.ones((2,bins)) * np.nan
                        else:
                            # starts = myFindLine(fname,[' 1 '])
                            starts = []
                            with open(fname, 'r') as f:
                                for n, line in enumerate(f):
                                    if line[0].isnumeric():
                                        starts.append(n+1)
                            # print(fname)
                            # print(starts)
                            val0 = np.loadtxt(fname, skiprows = starts[-1], usecols=(1,3)).T
                            # print('val0:',val0)
                            # tcols = len(pd.read_csv(fname,header=0,index_col=0,nrows=1).values[0])
                            # df = pd.read_csv(fname, header=0, index_col=0,usecols=[0,1,tcols])
                            # val0 = df.values.T
                            if 'edz' in datafile:
                                stage = 1
                            elif 'cdz' in datafile:
                                stage = 2
                            else:
                                stage = 3
                            brush = Bilayer(self.path)
                            PDa = [self.PDdirs.index(PDstr)]
                            Va = [self.Vdirs.index(Vstr)]
                            ans = brush.thermo(stage, ['v_bsurfz', 'f_D'], PDis=PDa,Vis=Va, avg=True)
                            bwzmax = ans[0, 0, 0, 0]
                            D = ans[0, 0, 1, 0]

                            if Z2D == True:
                                val0[0,:] = (val0[0,:] - bwzmax )/D
                            else:
                                val0[0,:] = (val0[0,:] - bwzmax)
                            val = val0
                    else:
                        val = np.ones((2, bins)) * np.nan
                    C[(i*2),:] = val[0,-bins:]
                    C[(i*2)+1,:] = val[1,-bins:]
                    i += 1
                B.append(C)
            A.append(B)
        D = np.asarray(A)
        return D

class Sims():
    def __init__(self, path):
        self.path = path
        self.topology = os.path.split(self.path.split('\Sims')[0])[1] + ' Brushes'
        Ndirs0 = os.listdir(self.path)
        Ndirs0.remove('PBB-ECS')
        Ns0 = [float(x.split('N=')[1]) for x in Ndirs0]
        self.Ndirs = [x for _,x in sorted(zip(Ns0,Ndirs0))]
        self.Ns = sorted(Ns0)
        Mdirs0 = list(filter(lambda x : 'M=' in x, os.listdir(r'{0}\{1}'.format(self.path,self.Ndirs[0]))))
        Ms0 = [float(x.split('M=')[1]) for x in Mdirs0]
        self.Mdirs = [x for _,x in sorted(zip(Ms0,Mdirs0))]
        self.Ms = sorted(Ms0)
        Xdirs0 = list(filter(lambda x : 'X=' in x, os.listdir(r'{0}\{1}\{2}'.format(self.path,self.Ndirs[0],self.Mdirs[0]))))
        Xs0 = [float(x.split('X=')[1]) for x in Xdirs0]
        self.Xdirs = [x for _,x in sorted(zip(Xs0,Xdirs0))]
        self.Xs = sorted(Xs0)
        A = []
        for Ndir in self.Ndirs:
            B = []
            for Mdir in self.Mdirs:
                C = []
                for Xdir in self.Xdirs:
                    dir = r'{0}\{1}\{2}\{3}'.format(self.path, Ndir, Mdir, Xdir)
                    C.append(Bilayer(dir))
                B.append(C)
            A.append(B)
        self.bilayers = A

    def info(self):
        info_array = []
        info_array.append(self.topology)
        info_array.append(self.Ndirs)
        info_array.append(self.Mdirs)
        info_array.append(self.Xdirs)
        info_array.append(self.bilayers[0][0][0].PDdirs)
        info_array.append(self.bilayers[0][0][0].Vdirs)
        return info_array

    def thermo_dic(self):
        print(self.bilayers[0][0][0].tdict)

    def mainVars(self,mvar,Nis=0,Mis=0,Xis=0,PDis=0,Vis=0):
        if Nis == 0:
            Nis = [i for i in range(len(self.Ndirs))]
        if Mis == 0:
            Mis = [i for i in range(len(self.Mdirs))]
        if Xis == 0:
            Xis = [i for i in range(len(self.Xdirs))]
        rows = 2
        if PDis == 0:
            PDis = [i for i in range(len(self.bilayers[0][0][0].PDdirs))]
        PDlength = len(PDis)
        if Vis == 0:
            Vis = [i for i in range(len(self.bilayers[0][0][0].Vdirs))]
        Vlength = len(Vis)
        ax = 1
        A = np.zeros((len(Nis),len(Mis),len(Xis),PDlength,Vlength,ax,rows))
        i = 0
        for Ni in Nis:
            j = 0
            for Mi in Mis:
                k = 0
                for Xi in Xis:
                    brush = self.bilayers[Ni][Mi][Xi]
                    var = brush.mainVars(mvar,PDis=PDis,Vis=Vis)
                    A[i, j, k] = var
                    k += 1
                j += 1
            i += 1
        B = np.ma.masked_invalid(A)
        return B

    def simVars(self,svar,Nis=0,Mis=0,Xis=0,PDis=0,Vis=0):
        if Nis == 0:
            Nis = [i for i in range(len(self.Ndirs))]
        if Mis == 0:
            Mis = [i for i in range(len(self.Mdirs))]
        if Xis == 0:
            Xis = [i for i in range(len(self.Xdirs))]
        rows = 2
        if PDis == 0:
            PDis = [i for i in range(len(self.bilayers[0][0][0].PDdirs))]
        PDlength = len(PDis)
        if Vis == 0:
            Vis = [i for i in range(len(self.bilayers[0][0][0].Vdirs))]
        Vlength = len(Vis)
        ax = 1
        A = np.zeros((len(Nis),len(Mis),len(Xis),PDlength,Vlength,ax,rows))
        i = 0
        for Ni in Nis:
            j = 0
            for Mi in Mis:
                k = 0
                for Xi in Xis:
                    brush = self.bilayers[Ni][Mi][Xi]
                    var = brush.simVars(svar,PDis=PDis,Vis=Vis)
                    A[i, j, k] = var
                    k += 1
                j += 1
            i += 1
        B = np.ma.masked_invalid(A)
        return B

    def thermo(self,stage, tvars, steps = 40,Nis=0,Mis=0,Xis=0,PDis=0,Vis=0, avg = True):
        if Nis == 0:
            Nis = [i for i in range(len(self.Ndirs))]
        if Mis == 0:
            Mis = [i for i in range(len(self.Mdirs))]
        if Xis == 0:
            Xis = [i for i in range(len(self.Xdirs))]
        if avg == True:
            rows = 2
        else:
            rows = steps
        if PDis == 0:
            PDis = [i for i in range(len(self.bilayers[0][0][0].PDdirs))]
        PDlength = len(PDis)
        if Vis == 0:
            Vis = [i for i in range(len(self.bilayers[0][0][0].Vdirs))]
        Vlength = len(Vis)
        if len(tvars) == 1:
            ax = 1
        else:
            ax  = (len(tvars)-1)*2
        A = np.zeros((len(Nis),len(Mis),len(Xis),PDlength,Vlength,ax,rows))
        i = 0
        for Ni in Nis:
            j = 0
            for Mi in Mis:
                k = 0
                for Xi in Xis:
                    brush = self.bilayers[Ni][Mi][Xi]
                    temp = brush.thermo(stage,tvars, steps=steps, PDis=PDis,Vis=Vis, avg=avg)
                    A[i,j,k] = temp
                    k += 1
                j += 1
            i += 1
        B = np.ma.masked_invalid(A)
        return B
    def thermoNM(self,stage,Nis=0,Mis=0,Xis=0,PDis=0,Vis=0):
        if Nis == 0:
            Nis = [i for i in range(len(self.Ndirs))]
        if Mis == 0:
            Mis = [i for i in range(len(self.Mdirs))]
        if Xis == 0:
            Xis = [i for i in range(len(self.Xdirs))]
        rows = 2
        if PDis == 0:
            PDis = [i for i in range(len(self.bilayers[0][0][0].PDdirs))]
        PDlength = len(PDis)
        if Vis == 0:
            Vis = [i for i in range(len(self.bilayers[0][0][0].Vdirs))]
        Vlength = len(Vis)
        ax = 2
        A = np.zeros((len(Nis),len(Mis),len(Xis),PDlength,Vlength,ax,rows))
        i = 0
        for Ni in Nis:
            j = 0
            for Mi in Mis:
                k = 0
                for Xi in Xis:
                    brush = self.bilayers[Ni][Mi][Xi]
                    temp = brush.thermoNM(stage, PDis=PDis,Vis=Vis)
                    A[i,j,k] = temp
                    k += 1
                j += 1
            i += 1
        B = np.ma.masked_invalid(A)
        return B

    def COF(self,PvS):
        # PvS = self.thermo(3, ['v_srate', 'pxz', 'pzz'], Nis=Nis, Mis=Mis, Xis=Xis, Pis=Pis, Vis=Vis, steps=steps, avg=True)
        dims = (PvS.shape[0],PvS.shape[1],PvS.shape[2],PvS.shape[3],PvS.shape[4],2,2)
        FvS = np.zeros(dims)
        for i in range(PvS.shape[0]):
            for j in range(PvS.shape[1]):
                for k in range(PvS.shape[2]):
                    for l in range(PvS.shape[3]):
                        for m in range(PvS.shape[4]):
                            gamma_dot = ufloat(PvS[i, j, k, l, m, 0, 0], PvS[i, j, k, l, m, 0, 1])
                            pxz = ufloat(PvS[i, j, k, l, m, 1, 0], PvS[i, j, k, l, m, 1, 1])
                            pzz = ufloat(PvS[i, j, k, l, m, 3, 0], PvS[i, j, k, l, m, 3, 1])
                            cof = - pxz / pzz
                            FvS[i, j, k, l, m, 0, 0] = gamma_dot.nominal_value
                            FvS[i, j, k, l, m, 0, 1] = gamma_dot.std_dev
                            FvS[i, j, k, l, m, 1, 0] = cof.nominal_value
                            FvS[i, j, k, l, m, 1, 1] = cof.std_dev
        B = np.ma.masked_invalid(FvS)
        return B
    def effVisc(self,PvS):
        # PvS = self.thermo(3, ['v_srate', 'pxz', 'pzz'], Nis=Nis, Mis=Mis, Xis=Xis, Pis=Pis, Vis=Vis, steps=steps, avg=True)
        dims = (PvS.shape[0],PvS.shape[1],PvS.shape[2],PvS.shape[3],PvS.shape[4],2,2)
        VvS = np.zeros(dims)
        for i in range(PvS.shape[0]):
            for j in range(PvS.shape[1]):
                for k in range(PvS.shape[2]):
                    for l in range(PvS.shape[3]):
                        for m in range(PvS.shape[4]):
                            gamma_dot = ufloat(PvS[i, j, k, l, m, 0, 0], PvS[i, j, k, l, m, 0, 1])
                            pxz = ufloat(PvS[i, j, k, l, m, 1, 0], PvS[i, j, k, l, m, 1, 1])
                            visc = - pxz / gamma_dot
                            VvS[i, j, k, l, m, 0, 0] = gamma_dot.nominal_value
                            VvS[i, j, k, l, m, 0, 1] = gamma_dot.std_dev
                            VvS[i, j, k, l, m, 1, 0] = visc.nominal_value
                            VvS[i, j, k, l, m, 1, 1] = visc.std_dev
        B = np.ma.masked_invalid(VvS)
        return B
    def profiles(self,datafiles,Nis=0,Mis=0,Xis=0,PDis=0,Vis=0,bins=1000, Z2D = True):
        if Nis == 0:
            Nis = [i for i in range(len(self.Ndirs))]
        if Mis == 0:
            Mis = [i for i in range(len(self.Mdirs))]
        if Xis == 0:
            Xis = [i for i in range(len(self.Xdirs))]

        if PDis == 0:
            PDis = [i for i in range(len(self.bilayers[0][0][0].PDdirs))]
        PDlength = len(PDis)
        if Vis == 0:
            Vis = [i for i in range(len(self.bilayers[0][0][0].Vdirs))]
        Vlength = len(Vis)
        trows = bins
        A = np.zeros((len(Nis),len(Mis),len(Xis),PDlength,Vlength,len(datafiles)*2,trows))
        i = 0
        for Ni in Nis:
            j = 0
            for Mi in Mis:
                k = 0
                for Xi in Xis:
                    brush = self.bilayers[Ni][Mi][Xi]
                    A[i, j, k] = brush.profiles(datafiles=datafiles, PDis=PDis, Vis=Vis,bins=bins, Z2D=Z2D)
                    k += 1
                j += 1
            i += 1
        B = np.ma.masked_invalid(A)
        return B

    def brush_heights(self,DPvZ):
        dims = (DPvZ.shape[0], DPvZ.shape[1], DPvZ.shape[2], DPvZ.shape[3], DPvZ.shape[4],1 ,2)
        B = np.zeros(dims)
        for i in range(DPvZ.shape[0]):
            for j in range(DPvZ.shape[1]):
                for k in range(DPvZ.shape[2]):
                    for l in range(DPvZ.shape[3]):
                        for m in range(DPvZ.shape[4]):
                            zs = DPvZ[i,j,k,l,m,0,:]
                            dps = DPvZ[i,j,k,l,m,1,:]
                            B[i,j,k,l,m,0,0]= myBrushHeight(zs,dps)
                            B[i, j, k, l, m,0,1] = np.nan
        B = np.ma.masked_invalid(B)
        return B

    def interpenetrations(self,DPvZ):
        #DPvZ = self.profiles(datafiles, Nis=Nis, Mis=Mis, Xis=Xis, Pis=Pis, Vis=Vis, bins=bins, Z2D=False)
        dims = (DPvZ.shape[0], DPvZ.shape[1], DPvZ.shape[2], DPvZ.shape[3], DPvZ.shape[4], 1, 2)
        B = np.zeros(dims)
        for i in range(DPvZ.shape[0]):
            for j in range(DPvZ.shape[1]):
                for k in range(DPvZ.shape[2]):
                    for l in range(DPvZ.shape[3]):
                        for m in range(DPvZ.shape[4]):
                            zs = DPvZ[i, j, k, l, m, 0, :]
                            bbdp = DPvZ[i, j, k, l, m, 1, :]
                            tbdp = DPvZ[i, j, k, l, m, 3, :]
                            if np.trapz(zs,bbdp) == 0.0:
                                B[i, j, k, l, m, 0, 0] = np.nan
                            else:
                                B[i, j, k, l, m, 0, 0] = np.trapz(bbdp * tbdp, zs)
                            B[i, j, k, l, m, 0, 1] = np.nan
        B = np.ma.masked_invalid(B)
        return B
    def overlap(self,DPvZ):
        #DPvZ = self.profiles(datafiles, Nis=Nis, Mis=Mis, Xis=Xis, Pis=Pis, Vis=Vis, bins=bins, Z2D=False)
        dims = (DPvZ.shape[0], DPvZ.shape[1], DPvZ.shape[2], DPvZ.shape[3], DPvZ.shape[4], 1, 2)
        B = np.zeros(dims)
        for i in range(DPvZ.shape[0]):
            for j in range(DPvZ.shape[1]):
                for k in range(DPvZ.shape[2]):
                    for l in range(DPvZ.shape[3]):
                        for m in range(DPvZ.shape[4]):
                            zs = DPvZ[i, j, k, l, m, 0, :]
                            bbdp = DPvZ[i, j, k, l, m, 1, :]
                            tbdp = DPvZ[i, j, k, l, m, 3, :]
                            abdp = bbdp+tbdp
                            start = False
                            end = False
                            si = 0
                            ei = 1
                            for n in range(abdp.shape[0]):
                                if abdp[n] != 0.0 and start is False:
                                    start = True
                                    si = n
                                if abdp[n] == 0.0 and (n-si > 20) and start is True and end is False:
                                    end = True
                                    ei = n
                            zc = zs[si:ei]
                            bbdpc = bbdp[si:ei]
                            tbdpc = tbdp[si:ei]
                            abdpc = bbdpc + tbdpc
                            mi = round(zc.shape[0]/2)
                            mid_den = abdpc[round(abdpc.shape[0]/2)]
                            if np.trapz(bbdpc,zc) == 0.0 or (si == 0 and ei == 1):
                                B[i, j, k, l, m, 0, 0] = np.nan
                            else:
                                B[i, j, k, l, m, 0, 0] = np.trapz(bbdp * tbdp, zs)
                            B[i, j, k, l, m, 0, 1] = np.nan
        B = np.ma.masked_invalid(B)
        return B
    def NMFHS(self,DPvZ):
        #DPvZ = self.profiles(datafiles, Nis=Nis, Mis=Mis, Xis=Xis, Pis=Pis, Vis=Vis, bins=bins, Z2D=False)
        dims = (DPvZ.shape[0], DPvZ.shape[1], DPvZ.shape[2], DPvZ.shape[3], DPvZ.shape[4], 1, 2)
        B = np.zeros(dims)
        for i in range(DPvZ.shape[0]):
            for j in range(DPvZ.shape[1]):
                for k in range(DPvZ.shape[2]):
                    for l in range(DPvZ.shape[3]):
                        for m in range(DPvZ.shape[4]):
                            zs = DPvZ[i, j, k, l, m, 0, :]
                            bbdp = DPvZ[i, j, k, l, m, 1, :]
                            tbdp = DPvZ[i, j, k, l, m, 3, :]
                            abdp = bbdp+tbdp
                            start = False
                            end = False
                            si = 0
                            ei = 1
                            for n in range(abdp.shape[0]):
                                if abdp[n] != 0.0 and start is False:
                                    start = True
                                    si = n
                                if abdp[n] == 0.0 and (n-si > 20) and start is True and end is False:
                                    end = True
                                    ei = n
                            zc = zs[si:ei]
                            bbdpc = bbdp[si:ei]
                            tbdpc = tbdp[si:ei]
                            abdpc = bbdpc + tbdpc
                            mi = round(zc.shape[0]/2)
                            mid_den = abdpc[round(abdpc.shape[0]/2)]
                            if np.trapz(bbdpc,zc) == 0.0 or (si == 0 and ei == 1):
                                B[i, j, k, l, m, 0, 0] = np.nan
                            else:
                                B[i, j, k, l, m, 0, 0] = np.trapz(bbdpc[mi:], zc[mi:])
                            B[i, j, k, l, m, 0, 1] = np.nan
        B = np.ma.masked_invalid(B)
        return B
    def FMFHS(self,DPvZ):
        #DPvZ = self.profiles(datafiles, Nis=Nis, Mis=Mis, Xis=Xis, Pis=Pis, Vis=Vis, bins=bins, Z2D=False)
        dims = (DPvZ.shape[0], DPvZ.shape[1], DPvZ.shape[2], DPvZ.shape[3], DPvZ.shape[4], 1, 2)
        B = np.zeros(dims)
        for i in range(DPvZ.shape[0]):
            for j in range(DPvZ.shape[1]):
                for k in range(DPvZ.shape[2]):
                    for l in range(DPvZ.shape[3]):
                        for m in range(DPvZ.shape[4]):
                            zs = DPvZ[i, j, k, l, m, 0, :]
                            bbdp = DPvZ[i, j, k, l, m, 1, :]
                            tbdp = DPvZ[i, j, k, l, m, 3, :]
                            abdp = bbdp+tbdp
                            start = False
                            end = False
                            si = 0
                            ei = 1
                            for n in range(abdp.shape[0]):
                                if abdp[n] != 0.0 and start is False:
                                    start = True
                                    si = n
                                if abdp[n] == 0.0 and (n-si > 20) and start is True and end is False:
                                    end = True
                                    ei = n
                            zc = zs[si:ei]
                            bbdpc = bbdp[si:ei]
                            tbdpc = tbdp[si:ei]
                            abdpc = bbdpc + tbdpc
                            mi = round(zc.shape[0]/2)
                            mid_den = abdpc[round(abdpc.shape[0]/2)]
                            if np.trapz(bbdpc,zc) == 0.0 or (si == 0 and ei == 1):
                                B[i, j, k, l, m, 0, 0] = np.nan
                            else:
                                B[i, j, k, l, m, 0, 0] = np.trapz(bbdpc[mi:], zc[mi:])/np.trapz(bbdpc, zc)
                            B[i, j, k, l, m, 0, 1] = np.nan
        B = np.ma.masked_invalid(B)
        return B
    def mid_den(self,DPvZ):
        #DPvZ = self.profiles(datafiles, Nis=Nis, Mis=Mis, Xis=Xis, Pis=Pis, Vis=Vis, bins=bins, Z2D=False)
        dims = (DPvZ.shape[0], DPvZ.shape[1], DPvZ.shape[2], DPvZ.shape[3], DPvZ.shape[4], 1, 2)
        B = np.zeros(dims)
        for i in range(DPvZ.shape[0]):
            for j in range(DPvZ.shape[1]):
                for k in range(DPvZ.shape[2]):
                    for l in range(DPvZ.shape[3]):
                        for m in range(DPvZ.shape[4]):
                            zs = DPvZ[i, j, k, l, m, 0, :]
                            bbdp = DPvZ[i, j, k, l, m, 1, :]
                            tbdp = DPvZ[i, j, k, l, m, 3, :]
                            abdp = bbdp+tbdp
                            start = False
                            end = False
                            si = 0
                            ei = 1
                            for n in range(abdp.shape[0]):
                                if abdp[n] != 0.0 and start is False:
                                    start = True
                                    si = n
                                if abdp[n] == 0.0 and (n-si > 20) and start is True and end is False:
                                    end = True
                                    ei = n
                            zc = zs[si:ei]
                            bbdpc = bbdp[si:ei]
                            tbdpc = tbdp[si:ei]
                            abdpc = bbdpc + tbdpc
                            mi = round(zc.shape[0]/2)
                            mid_den = abdpc[round(abdpc.shape[0]/2)]
                            if np.trapz(bbdpc, zc) == 0.0 or (si == 0 and ei == 1):
                                B[i, j, k, l, m, 0, 0] = np.nan
                            else:
                                B[i, j, k, l, m, 0, 0] = mid_den
                            B[i, j, k, l, m, 0, 1] = np.nan
        B = np.ma.masked_invalid(B)
        return B


    def Rg2x(self,RvS):
        dims = RvS.shape
        R2vS = np.zeros(dims)
        for i in range(RvS.shape[0]):
            for j in range(RvS.shape[1]):
                for k in range(RvS.shape[2]):
                    for l in range(RvS.shape[3]):
                        R0x_vs = RvS[i, j, k, l, :, 1, 0]
                        R0x_i = np.where(R0x_vs == np.nanmin(R0x_vs))[0][0]
                        R0x = ufloat(RvS[i, j, k, l, R0x_i, 1, 0], RvS[i, j, k, l, R0x_i, 1, 1])
                        for m in range(RvS.shape[4]):
                            gamma_dot = ufloat(RvS[i, j, k, l, m, 0, 0], RvS[i, j, k, l, m, 0, 1])
                            Rgx = ufloat(RvS[i, j, k, l, m, 1, 0], RvS[i, j, k, l, m, 1, 1])
                            Rg2x_Rg20x = (Rgx**2 / R0x**2)
                            R2vS[i, j, k, l, m, 0, 0] = gamma_dot.nominal_value
                            R2vS[i, j, k, l, m, 0, 1] = gamma_dot.std_dev
                            R2vS[i, j, k, l, m, 1, 0] = Rg2x_Rg20x.nominal_value
                            R2vS[i, j, k, l, m, 1, 1] = Rg2x_Rg20x.std_dev
        B = np.ma.masked_invalid(R2vS)
        return B

    def srate_crit(self,R2xvS):
        dims = (R2xvS.shape[0], R2xvS.shape[1], R2xvS.shape[2], R2xvS.shape[3], 1)
        srates = np.zeros(dims)
        for i in range(R2xvS.shape[0]):
            for j in range(R2xvS.shape[1]):
                for k in range(R2xvS.shape[2]):
                    for l in range(R2xvS.shape[3]):
                        srates[i,j,k,l] = myCrit_Srate(R2xvS[i,j,k,l,:,0,0],R2xvS[i,j,k,l,:,1,0])
        B = np.ma.masked_invalid(srates)
        return B

    def Weissenberg(self,R2xvS,src):
        R2xvW = np.copy(R2xvS)
        for i in range(R2xvS.shape[0]):
            for j in range(R2xvS.shape[1]):
                for k in range(R2xvS.shape[2]):
                    for l in range(R2xvS.shape[3]):
                        srate_crit = src[i, j, k, l]
                        for m in range(R2xvS.shape[4]):
                            R2xvW[i,j,k,l,:,0,0] = ((R2xvS[i,j,k,l,:,0,0]) / srate_crit)
        B = np.ma.masked_invalid(R2xvW)
        return B

    def mask(self,FvS,src):
        B = np.copy(FvS)
        for i in range(FvS.shape[0]):
            for j in range(FvS.shape[1]):
                for k in range(FvS.shape[2]):
                    for l in range(FvS.shape[3]):
                        srate_crit = src[i, j, k, l]
                        for m in range(FvS.shape[4]):
                            if FvS[i,j,k,l,m,0,0] < (srate_crit)*1.00001:
                                B[i,j,k,l,m,0,0] = np.nan
                                B[i,j,k,l,m,1,0] = np.nan
                                B[i, j, k, l, m, 0, 1] = np.nan
                                B[i, j, k, l, m, 1, 1] = np.nan

        FvS_masked = np.ma.masked_invalid(B)
        return FvS_masked

if __name__ == '__main__':

    t0 = time.perf_counter()
    print('Hello')
    print(sys.version)

    # LL = myLastLine(r'X:\PhD-WD\PaperA1\Linear\Sims\N=20\M=230\X=0\D=8\PBB-ECS\equil.log')
    #
    # print(LL)
    # brush = Bilayer(r'X:\TrimmedSims\PaperA1X\Wall\Linear\Sims\N=30\M=46\X=0')
    # print(brush.path)
    # X = brush.simVars('surfcov',PDis=[0,1,2],Vis=[0])
    # print(X)



    Topologies = []
    Linear = Sims(r'X:\PhD-WD\Paper09\Linear\Sims')
    #
    # Topologies.append(Ring)
    Topologies.append(Linear)
    Topologies[0].thermo_dic()
    for Topology in Topologies:
        print(Topology.info())

    print(Topologies[0].bilayers)

    for Topology in Topologies:
        iDPSds = Topology.profiles(['bbeads_sdz', 'tbeads_sdz'], Nis=[9], Mis=[5], Xis=[0], PDis=0, Vis=[5], Z2D=False)
        MDds = Topology.mid_den(iDPSds)
        print(MDds)







    t1 = time.perf_counter()
    print('Time Elapsed: {0:.2f} s'.format(t1-t0))







    # X = Topologies[0].simVars(svar='surfcov', Nis=[0], Mis=[0], Xis=[0], PDis=[0], Vis=[0,1])
    # print(X)

    # eDPS = []
    # RhovH = []
    # eDPSd = Topology.profiles(['bbeads_edz'], Nis=[0], Mis=0, Xis=[0], PDis=[0], Vis=[0], Z2D=False)
    # print(eDPSd.shape)
    # eDPSr = reorder_data(eDPSd)
    # eDPS.append(eDPSr[0, :, 0, 0, 0, :, :])
    # print(eDPS[0].shape)
    #
    # fig, eDPSax = plt.subplots()
    # axs = myAxs(datas=eDPS, ax=eDPSax, xlim=[0, 90],xlabel='$z$', ylabel=r'$\phi(z)$')
    # plt.show()



    # Hsr = Topology.brush_heights(eDPSd)
    # # Rhosd = Topology.thermo(1, tvars=['v_surfcov'], Nis=[0], Mis=[0], Xis=[0], PDis=[0], Vis=[0])
    # Rhosd = Topology.simVars(svar='surfcov', Nis=[0], Mis=[0], Xis=[0], PDis=[0], Vis=[0])
    # RhovHr = np.ma.concatenate((Rhosd, Hsr), axis=-2)
    # RhovH.append(RhovHr[:, :, :, :, :, :, :])


    r'''

    # Equilibrium
    DPS = []
    for Topology in Topologies:
        DPSd = Topology.profiles(['bbeads_edz'], Nis=[0], Mis=[0], Xis=[0], PDis=[0], Vis=[0], Z2D=False)
        DPS.append(DPSd[:,:,0,0,0,:,:])
        print(DPSd)
    fig,ax = plt.subplots()
    Ns = [0]
    for N in Ns:
        ax.clear()
        DPSp = [DPS[i][N,:,:,:] for i in range(len(DPS))]
        axs = myAxs(DPSp,ax,llabels=(1,[Mdir for Mdir in Ring.Mdirs]),xlim=[0,90],xlabel='$z$', ylabel=r'$\phi(z)$')
        figurename = '1A-EquilProfiles-{0}.jpg'.format(Ring.Ndirs[N])
        print(figurename)
        plt.savefig(figurename)



    RhovH = []
    for Topology in Topologies:
        DPSd = Topology.profiles(['bbeads_edz'], Nis=0, Mis=0, Xis=[0], PDis=[0], Vis=[0], Z2D=False)
        Hsr = Topology.brush_heights(DPSd)
        Rhosd = Topology.thermo(1, tvars=['v_surfcov'], Nis=0, Mis=0, Xis=[0], PDis=[0], Vis=[0])
        RhovHr = np.ma.concatenate((Rhosd, Hsr), axis=-2)
        RhovH.append(RhovHr[:, :, 0, 0, 0, :, :])
    fig,ax = plt.subplots()
    RhovHp = RhovH
    axs = myErrorAxs(RhovHp, ax,llabels=(1,Ring.Ndirs),xlabel=r'$\rho_g$',ylabel=r'$h$',bars=False,log=(True,True),power_law=(0,0.333,'h',r'ρ'))
    figurename = '1B-HvsRhog.jpg'
    print(figurename)
    plt.savefig(figurename)

    Ns = [0,2]
    Ms = [0,2]
    PDs = [0,2]
    for N in Ns:
        for M in Ms:
            for PD in PDs:
                DPS = []
                for Topology in Topologies:
                    DPSd = Topology.profiles(['bbeads_cdz', 'abeads_cdz', 'tbeads_cdz'], Nis=[N], Mis=[M], Xis=[0], PDis=[PD], Vis=[0],Z2D=True)
                    DPS.append(DPSd[0, 0, 0, 0, 0, :, :])
                fig,ax = plt.subplots()
                DPSp = DPS
                myAxs(DPSp,ax,llabels = (0,['Ring','Linear']),xlim=[0,1] ,xlabel='$z/D$', ylabel=r'$\phi(z)$')
                figurename = '2A-CompProfiles-{0}-{1}-{2}.jpg'.format(Ring.Ndirs[N],Ring.Mdirs[M],Ring.bilayers[0][0][0].PDdirs[PD])
                print(figurename)
                plt.savefig(figurename)
                plt.close()

    # # Compression Curves
    Ns = [0]
    i = 0
    for N in Ns:
        PvD = []
        for Topology in Topologies:
            PvDd = Topology.thermo(2, ['v_D', 'pzz'], Nis=[N], Mis=0, Xis=[0], PDis=0, Vis=[0], avg=True)
            PvD.append(PvDd[0, :, 0, :, 0, :, :])
        fig, ax = plt.subplots()
        PvDp = PvD
        myErrorAxs(PvDp,ax,llabels=(1,Ring.Mdirs), xlabel='$D$', ylabel=r'$P$',log=(True,True), power_law=(1700,-3.5,'D','P'))
        figurename = '2B-PvsD-{0}.jpg'.format(Ring.Ndirs[i])
        print(figurename)
        plt.savefig(figurename)
        plt.close()


    # Interpenetration vs D
    Ns = [0]
    i = 0
    for N in Ns:
        IvD = []
        for Topology in Topologies:
            DPSd = Topology.profiles(['bbeads_cdz', 'tbeads_cdz'], Nis=[N], Mis=0, Xis=[0], PDis=0, Vis=[0], Z2D=True)
            Isd = Topology.interpenetrations(DPSd)
            Dsd = Topology.thermo(2, tvars=['v_D'], Nis=[N], Mis=0, Xis=[0], PDis=0, Vis=[0])
            IvDr = np.ma.concatenate((Dsd, Isd), axis=-2)
            IvD.append(IvDr[0, :, 0, :, 0, :, :])
        fig,ax = plt.subplots()
        IvDp = IvD
        myErrorAxs(IvDp,ax,llabels=(1,Ring.Mdirs),xlabel=r'D',ylabel=r'$I$',bars=False,log=(True,True),power_law=(0.08,-0.31,'I','D'))
        figurename = '2B-IvsD-{0}.jpg'.format(Ring.Ndirs[N])
        print(figurename)
        plt.savefig(figurename)
        plt.close()

    Ns = [0]
    Ms = [0]
    PDs = [0]
    Vs = [6]
    for N in Ns:
        for M in Ms:
            for PD in PDs:
                for V in Vs:
                    cps = []
                    for Topology in Topologies:
                        cdata = Topology.profiles(['temp_sz', 'abeads_sdz', 'velp_sz'], Nis=[N], Mis=[M], Xis=[0], PDis=[PD], Vis=[V], Z2D=True)
                        cps.append(cdata[0, 0, 0, 0, 0, :, :])
                    fig, ax = plt.subplots()
                    cpsp = cps
                    try:
                        name = '{0}-{1}-{2}-{3}'.format(Topologies[0].Ndirs[N], Topologies[0].Mdirs[M],
                                                        Topologies[0].bilayers[N][M][0].PDdirs[PD],
                                                        Topologies[0].bilayers[N][M][0].Vdirs[V])
                    except:
                        name = '{0}-{1}-{2}-V{3}'.format(Topologies[0].Ndirs[N], Topologies[0].Mdirs[M],
                                                         Topologies[0].bilayers[N][M][0].PDdirs[PD], V)
                    myAxs(cpsp, ax, llabels=(2, ['Temperature', 'Density', 'Velocity']), axis2=[0, 0, 1],
                         xlim=[0, 1],
                         xlabel='$z/D$', ylabel=r'$\phi(z), T$', y2label='V')
                    figurename = '3A-TDVProfiles-{0}.jpg'.format(name)
                    print(figurename)
                    plt.savefig(figurename)
                    plt.close()


    Ns = [0]
    PDs = [0]
    for N in Ns:
        for PD in PDs:
            RvS= []
            RvW = []
            PvS = []
            FvS = []
            FvS_m = []
            FvW = []
            FvW_m = []
            for Topology in Topologies:
                RvS_c = Topology.thermo(3, ['v_srate', 'v_aveRgx'], Nis=[N], Mis=0, Xis=[0], PDis=[PD], Vis=0, avg=True)
                Rg2xvS_c = Ring.Rg2x(RvS_c)
                RvS.append(Rg2xvS_c[:, :, :, :, :, :, :])
                src_c = Topology.srate_crit(Rg2xvS_c)
                Rg2xvW_c = Topology.Weissenberg(Rg2xvS_c, src_c)
                RvW.append(Rg2xvW_c[:, :, :, :, :, :, :])
                PvS_c = Topology.thermo(3, ['v_srate', 'pxz', 'pzz'], Nis=[N], Mis=0, Xis=[0], PDis=[PD], Vis=0,
                                        avg=True)
                PvS.append(PvS_c[:, :, :, :, :, :, :])
                FvS_c = Topology.COF(PvS_c)
                FvS.append(FvS_c[:, :, :, :, :, :, :])
                FvS_m_c = Ring.mask(FvS_c, src_c)
                FvS_m.append(FvS_m_c[:, :, :, :, :, :, :])
                FvW.append(Topology.Weissenberg(FvS_c, src_c))
                FvW_m.append(Topology.Weissenberg(FvS_m_c, src_c))
        name = '{0}-{1}'.format(Topologies[0].Ndirs[N], Topologies[0].bilayers[0][0][0].PDdirs[PD])
        fig, ax = plt.subplots()
        RvSp = [RvS[i][0, :, 0, 0, :, :, :] for i in range(len(RvS))]
        myErrorAxs(RvSp, ax, llabels=(1, Topologies[0].Mdirs), xlabel=r'$\.\gamma$', ylabel=r'$Rgx$', log=(True, True))
        figurename = '3B-Rg2xvsSrate-{0}.jpg'.format(name)
        print(figurename)
        plt.savefig(figurename)
        plt.close()
        fig, ax = plt.subplots()
        RvWp = [RvW[i][0, :, 0, 0, :, :, :] for i in range(len(RvW))]
        myErrorAxs(RvWp, ax, llabels=(1, Topologies[0].Mdirs), xlabel=r'$W$', ylabel=r'$Rgx$', log=(True, True))
        figurename = '3B-Rg2xvsW-{0}.jpg'.format(name)
        print(figurename)
        plt.savefig(figurename)
        plt.close()
        fig, ax = plt.subplots()
        PvSp = [PvS[i][0, :, 0, 0, :, :, :] for i in range(len(PvS))]
        myErrorAxs(PvSp, ax, llabels=(1, Topologies[0].Mdirs), xlabel=r'$\.\gamma$', ylabel=r'$Pxz$', axis2=[0, 1],
                   y2label=r'$Pzz$', log=(True, True))
        figurename = '3C-PvsSrate-{0}.jpg'.format(name)
        print(figurename)
        plt.savefig(figurename)
        plt.close()
        fig, ax = plt.subplots()
        FvSp = [FvS[i][0, :, 0, 0, :, :, :] for i in range(len(FvS))]
        myErrorAxs(FvSp, ax, llabels=(1, Topologies[0].Mdirs), xlabel=r'$\.\gamma$', ylabel=r'$\mu$', log=(True, True))
        figurename = '3D-CoFvsSrate-{0}.jpg'.format(name)
        print(figurename)
        plt.savefig(figurename)
        plt.close()
        fig, ax = plt.subplots()
        FvS_mp = [FvS_m[i][0, :, 0, 0, :, :, :] for i in range(len(FvS_m))]
        myErrorAxs(FvS_mp, ax, llabels=(1, Topologies[0].Mdirs), xlabel=r'$\.\gamma$', ylabel=r'$\mu$',
                   log=(True, True), power_law=(0.2,0.54,'CoF','S'))
        figurename = '3D-CoFvsSrate-masked-{0}.jpg'.format(name)
        print(figurename)
        plt.savefig(figurename)
        plt.close()
        fig, ax = plt.subplots()
        FvWp = [FvW[i][0, :, 0, 0, :, :, :] for i in range(len(FvS))]
        myErrorAxs(FvWp, ax, llabels=(1, Topologies[0].Mdirs), xlabel=r'$W$', ylabel=r'$\mu$', log=(True, True))
        figurename = '3D2-CoFvsW-{0}.jpg'.format(name)
        print(figurename)
        plt.savefig(figurename)
        plt.close()
        fig, ax = plt.subplots()
        FvW_mp = [FvW_m[i][0, :, 0, 0, :, :, :] for i in range(len(FvW_m))]
        myErrorAxs(FvW_mp, ax, llabels=(1, Topologies[0].Mdirs), xlabel=r'$W$', ylabel=r'$\mu$',
                   log=(True, True))
        figurename = '3D2-CoFvsW-masked-{0}.jpg'.format(name)
        print(figurename)
        plt.savefig(figurename)
        plt.close()
    '''


    r'''

    PDs = [5]
    Vs = [5]

    for PD in PDs:
        for V in Vs:
            FvM = []
            for Topology in Topologies:
                FvMd = FvS
                FvSr = reorder_data(FvSd,[0,1,2,3,4,5,6])
                DPSr = reorder_data(DPSd)


    '''





    r'''

    Ns = [0]
    PDs = [0]
    for N in Ns:
        for PD in PDs:
            RvS = []
            RvW = []
            for Topology in Topologies:
                RvS_c = Topology.thermo(3, ['v_srate', 'v_aveRgx'], Nis=[N], Mis=0, Xis=[0], PDis=[PD], Vis=0, avg=True)
                Rg2xvS_c = Ring.Rg2x(RvS_c)
                RvS.append(Rg2xvS_c[0, :, 0, 0, :, :, :])
                src_c = Topology.srate_crit(Rg2xvS_c)
                Rg2xvW_c = Topology.Weissenberg(Rg2xvS_c,src_c)
                RvW.append(Rg2xvW_c[0, :, 0, 0, :, :, :])
            fig, ax = plt.subplots()
            RvSp = RvS
            RvWp = RvW
            name = '{0}-{1}'.format(Topologies[0].Ndirs[N], Topologies[0].bilayers[0][0][0].PDdirs[PD])
            myErrorAxs(RvSp,ax, llabels=(1, Topologies[0].Mdirs), xlabel=r'$\.\gamma$', ylabel=r'$Rgx$', log=(True, True))
            figurename = '3B-Rg2xvsSrate-{0}.jpg'.format(name)
            print(figurename)
            plt.savefig(figurename)
            plt.close()
            fig, ax = plt.subplots()
            myErrorAxs(RvWp,ax, llabels=(1, Topologies[0].Mdirs), xlabel=r'$\.\gamma$', ylabel=r'$Rgx$', log=(True, True))
            figurename = '3B-Rg2xvsW-{0}.jpg'.format(name)
            print(figurename)

    for N in Ns:
        for PD in PDs:
            PvS = []
            for Topology in Topologies:
                PvS_c = Topology.thermo(3, ['v_srate', 'pxz', 'pzz'], Nis=[N], Mis=0, Xis=[0], PDis=[PD], Vis=0, avg=True)
                PvS.append(PvS_c[:, :, :, :, :, :, :])
            fig, ax = plt.subplots()
            PvSp = [PvS[i][0, :, 0, 0, :, :, :] for i in range(len(PvS))]
            name = '{0}-{1}'.format(Topologies[0].Ndirs[N], Topologies[0].bilayers[0][0][0].PDdirs[PD])
            figurename = '3C-PvsSrate-{0}.jpg'.format(name)
            myErrorAxs(PvSp,ax, llabels=(1, Topologies[0].Mdirs), xlabel=r'$\.\gamma$', ylabel=r'$Pxz$', axis2=[0, 1],
                            y2label=r'$Pzz$', log=(True, True))
            print(figurename)
            plt.savefig(figurename)


    print("Time Elapsed: {0:.2f} s".format(t1-t0))
    for N in Ns:
        for PD in PDs:
            FvS = []
            for i in range(len(Topologies)):
                FvS_c = Topology.COF(PvS[i])
                FvS.append(FvS_c[:, :, :, :, :, :, :])
            fig, ax = plt.subplots()
            FvSp = [FvS[i][0, :, 0, 0, :, :, :] for i in range(len(FvS))]
            name = '{0}-{1}'.format(Topologies[0].Ndirs[N], Topologies[0].bilayers[0][0][0].PDdirs[PD])
            # myErrorPlot(FvS, llabels=(1, Topologies[0].Mdirs), xlabel=r'$\.\gamma$', ylabel=r'$\mu$',
            #             fname='3D-CoFvsSrate', names=[name], log=(True, True))
            myErrorAxs(FvSp, ax, llabels=(1, Topologies[0].Mdirs), xlabel=r'$\.\gamma$', ylabel=r'$\mu$', log=(True, True))
            figurename = '3D-CoFvsSrate-{0}.jpg'.format(name)
            print(figurename)
            plt.savefig(figurename)

    for N in Ns:
        for PD in PDs:
            FvS_m = []
            for i in range(len(Topologies)):
                FvS_m_c = Ring.mask(FvS[i], src_c)
                FvS_m.append(FvS_m_c[:, :, :, :, :, :, :])
            fig, ax = plt.subplots()
            FvS_mp = [FvS_m[i][0, :, 0, 0, :, :, :] for i in range(len(FvS))]
            name = '{0}-{1}'.format(Topologies[0].Ndirs[N], Topologies[0].bilayers[0][0][0].PDdirs[PD])
            myErrorAxs(FvS_mp,ax, llabels=(1, Topologies[0].Mdirs), xlabel=r'$\.\gamma$', ylabel=r'$\mu$', log=(True, True))
            figurename = '3D-CoFvsSrate-masked-{0}.jpg'.format(name)
            print(figurename)
            plt.savefig(figurename)

    for N in Ns:
        for PD in PDs:
            FvW = []
            for Topology in Topologies:
                FvS_c = Ring.mask(FvS[i], src_c)
                FvW.append(Topology.Weissenberg(FvS_c, src_c))
            myErrorPlot(FvW, llabels=(1, Ring.Mdirs), xlabel=r'$W$', ylabel=r'$\mu$', fname='3D2-CoFvsW',
                        names=[name], log=(True, True))

#STIL OOOOOOOOOLD
    for N in Ns:
        for PD in PDs:
            FvW = []
            for Topology in Topologies:
                RvS_c = Topology.thermo(3, ['v_srate', 'v_aveRgx'], Nis=[N], Mis=0, Xis=[0], PDis=[PD], Vis=0, avg=True)
                Rg2xvS_c = Topology.Rg2x(RvS_c)
                src_c = Topology.srate_crit(Rg2xvS_c)
                PvS_c = Topology.thermo(3, ['v_srate', 'pxz', 'pzz'], Nis=[N], Mis=0, Xis=[0], PDis=[PD], Vis=0,avg=True)
                FvS_c = Topology.COF(PvS_c)
                FvW.append(Topology.Weissenberg(FvS_c, src_c))
            myErrorPlot(FvW, llabels=(1, Ring.Mdirs), xlabel=r'$W$', ylabel=r'$\mu$', fname='3D2-CoFvsW',
                        names=[name], log=(True, True))

    t1 = time.perf_counter()
    print("Time Elapsed: {0:.2f} s".format(t1-t0))

    '''

    r'''
    # One Go Shearing
    Ns = [0]
    PDs = [0]
    for N in Ns:
        for PD in PDs:
            name = '{0}-{1}'.format(Ring.Ndirs[N],Ring.bilayers[0][0][0].PDdirs[PD])
            RvS_c = Ring.thermo(3, ['v_srate', 'v_aveRgx'], Nis=[N],Mis=0,Xis=[0],PDis=[PD],Vis=0 ,avg=True)
            RvS_l = Linear.thermo(3, ['v_srate', 'v_aveRgx'], Nis=[N], Mis=0, Xis=[0], PDis=[PD], Vis=0, avg=True)
            Rg2xvS_c = Ring.Rg2x(RvS_c)
            Rg2xvS_l = Linear.Rg2x(RvS_l)
            cRvS = Rg2xvS_c[:,:,0,0,:,:,:]
            lRvS = Rg2xvS_l[:,:,0,0,:,:,:]
            myErrorPlot([cRvS,lRvS],llabels=(1,Ring.Mdirs), xlabel=r'$\.\gamma$', ylabel=r'$Rgx$', fname='3B-Rg2xvsSrate',names=[name], log=(True,True))
            src_c = Ring.srate_crit(Rg2xvS_c)
            src_l = Linear.srate_crit(Rg2xvS_l)
            Rg2xvW_c = Ring.Weissenberg(Rg2xvS_c,src_c)
            Rg2xvW_l = Linear.Weissenberg(Rg2xvS_l, src_l)
            cRvW = Rg2xvW_c[:, :, 0, 0, :, :, :]
            lRvW = Rg2xvW_l[:, :, 0, 0, :, :, :]
            myErrorPlot([cRvW, lRvW], llabels=(1, Ring.Mdirs), xlabel=r'$W$', ylabel=r'$Rgx$',fname='3B2-Rg2xvsW', names=[name], log=(True, True))
            PvS_c = Ring.thermo(3, ['v_srate', 'pxz', 'pzz'], Nis=[N], Mis=0, Xis=[0], PDis=[PD], Vis=0, avg=True)
            PvS_l = Linear.thermo(3,['v_srate','pxz','pzz'],Nis=[N],Mis=0,Xis=[0],PDis=[PD],Vis=0,avg = True)
            cPvS = PvS_c[:,:,0,0,:,:,:]
            lPvS = PvS_l[:,:,0,0,:,:,:]
            myErrorPlot([cPvS,lPvS],llabels=(1,Ring.Mdirs), xlabel=r'$\.\gamma$', ylabel=r'$Pxz$',axis2=[0,1],y2label=r'$Pzz$', fname='3C-PvsSrate',names=[name], log=(True,True))
            FvS_c = Ring.COF(PvS_c)
            FvS_l = Linear.COF(PvS_l)
            cFvS = FvS_c[:, :, 0, 0, :, :, :]
            lFvS = FvS_l[:, :, 0, 0, :, :, :]
            myErrorPlot([cFvS, lFvS], llabels=(1, Ring.Mdirs), xlabel=r'$\.\gamma$',ylabel=r'$\mu$', fname='3D-CoFvsSrate', names=[name], log=(True, True))
            FvW_c = Ring.Weissenberg(FvS_c,src_c)
            FvW_l = Linear.Weissenberg(FvS_l,src_l)
            cFvW = FvW_c[:, :, 0, 0, :, :, :]
            lFvW = FvW_l[:, :, 0, 0, :, :, :]
            myErrorPlot([cFvW, lFvW], llabels=(1, Ring.Mdirs), xlabel=r'$W$',ylabel=r'$\mu$', fname='3D2-CoFvsW', names=[name], log=(True, True))
            FvS_m_c = Ring.mask(FvS_c,src_c)
            FvS_m_l = Linear.mask(FvS_l,src_l)
            cFvS_m = FvS_m_c[:, :, 0, 0, :, :, :]
            lFvS_m = FvS_m_l[:, :, 0, 0, :, :, :]
            myErrorPlot([cFvS_m, lFvS_m], llabels=(1, Ring.Mdirs), xlabel=r'$\.\gamma$', ylabel=r'$\mu$',fname='3D-CoFvsSrate-masked', names=[name], log=(True, True))


    '''



    #erbas15
    r'''
    f2al = Run(r'H:\OneDrive - Imperial College London\PhD\PhD-Sims\Paper21-Dev\erbas15\local\PBB-ECS-f2al')
    f2aR = Run(r'H:\OneDrive - Imperial College London\PhD\PhD-Sims\Paper21-Dev\erbas15\local\PBB-ECS-f2aR')
    f2bl = Run(r'H:\OneDrive - Imperial College London\PhD\PhD-Sims\Paper21-Dev\erbas15\local\PBB-ECS-f2bl')
    f2bR = Run(r'H:\OneDrive - Imperial College London\PhD\PhD-Sims\Paper21-Dev\erbas15\local\PBB-ECS-f2bR')

    print(f2al.tdict)

    PvS_ra = f2aR.thermo(3, ['v_Vwall', 'pxz'], Vis=0, avg=True)
    PvS_la = f2al.thermo(3, ['v_Vwall', 'pxz'], Vis=0, avg=True)
    PvS_rb = f2bR.thermo(3, ['v_Vwall', 'pxz'], Vis=0, avg=True)
    PvS_lb = f2bl.thermo(3, ['v_Vwall', 'pxz'], Vis=0, avg=True)
    PvS_r = np.hstack((PvS_ra,PvS_rb))
    PvS_l = np.hstack((PvS_la,PvS_lb))


    D_r = f2aR.thermo(3, ['v_D'], Vis=[2], avg=True)
    D_l = f2al.thermo(3, ['v_D'], Vis=[2], avg=True)
    Axy = (f2aR.xhi * f2aR.yhi)

    myErrorPlot([abs(PvS_r), abs(PvS_l)], llabels=(2, ['2a-N={0}-M={1}-D={2}'.format(f2al.N,f2al.M,f2al.Dcomp), '2b-N={0}-M={1}-D={2}'.format(f2bl.N,f2bl.M,f2bl.Dcomp)]), xlabel=r'$V$', ylabel=r'$Pxz$',fname='0A-FFvsV-D={0}'.format(f2al.Dcomp), names=['f2a-Pxz'], log=(True, True))

    PvS_r[:,1,:] = PvS_r[:,1,:] * Axy
    PvS_l[:,1,:] = PvS_l[:,1,:] * Axy
    PvS_r[:,3,:] = PvS_r[:,3,:] * Axy
    PvS_l[:,3,:] = PvS_l[:,3,:] * Axy

    myErrorPlot([abs(PvS_r), abs(PvS_l)], llabels=(2, ['2a-N={0}-M={1}-D={2}'.format(f2al.N,f2al.M,f2al.Dcomp), '2b-N={0}-M={1}-D={2}'.format(f2bl.N,f2bl.M,f2bl.Dcomp)]), xlabel=r'$V$', ylabel=r'$Pxz$',fname='0A-FFvsV-D={0}'.format(f2al.Dcomp), names=['f2a-PxzAxy'], log=(True, True))

    PvS_ra = f2aR.thermo(3, ['v_Vwall', 'v_Ps_t'],  Vis=0, avg=True)
    PvS_la = f2al.thermo(3, ['v_Vwall', 'v_Ps_t'],  Vis=0, avg=True)
    PvS_rb = f2bR.thermo(3, ['v_Vwall', 'v_Ps_t'],  Vis=0, avg=True)
    PvS_lb = f2bl.thermo(3, ['v_Vwall', 'v_Ps_t'],  Vis=0, avg=True)
    PvS_r = np.hstack((PvS_ra,PvS_rb))
    PvS_l = np.hstack((PvS_la,PvS_lb))

    myErrorPlot([abs(PvS_r), abs(PvS_l)], llabels=(2, ['2a-N={0}-M={1}-D={2}'.format(f2al.N,f2al.M,f2al.Dcomp), '2b-N={0}-M={1}-D={2}'.format(f2bl.N,f2bl.M,f2bl.Dcomp)]), xlabel=r'$V$', ylabel=r'$Pxz$', fname='0A-FFvsV-D={0}'.format(f2al.Dcomp), names=['f2a-Ps_t'], log=(True, True))

    PvS_r[:,1,:] = PvS_r[:,1,:] * Axy
    PvS_l[:,1,:] = PvS_l[:,1,:] * Axy
    PvS_r[:,3,:] = PvS_r[:,3,:] * Axy
    PvS_l[:,3,:] = PvS_l[:,3,:] * Axy
    myErrorPlot([abs(PvS_r), abs(PvS_l)], llabels=(2, ['2a-N={0}-M={1}-D={2}'.format(f2al.N,f2al.M,f2al.Dcomp), '2b-N={0}-M={1}-D={2}'.format(f2bl.N,f2bl.M,f2bl.Dcomp)]), xlabel=r'$V$', ylabel=r'$Pxz$',fname='0A-FFvsV-D={0}'.format(f2al.Dcomp), names=['f2a-Ps_tAxy'], log=(True, True))

    PvS_ra = f2aR.thermo(3, ['v_Vwall', 'c_ftwall[1]'],  Vis=0, avg=True)
    PvS_la = f2al.thermo(3, ['v_Vwall', 'c_ftwall[1]'],  Vis=0, avg=True)
    PvS_rb = f2bR.thermo(3, ['v_Vwall', 'c_ftwall[1]'],  Vis=0, avg=True)
    PvS_lb = f2bl.thermo(3, ['v_Vwall', 'c_ftwall[1]'],  Vis=0, avg=True)
    PvS_r = np.hstack((PvS_ra,PvS_rb))
    PvS_l = np.hstack((PvS_la,PvS_lb))
    print(PvS_r.shape)

    myErrorPlot([abs(PvS_r), abs(PvS_l)], llabels=(2, ['2a-N={0}-M={1}-D={2}'.format(f2al.N,f2al.M,f2al.Dcomp), '2b-N={0}-M={1}-D={2}'.format(f2bl.N,f2bl.M,f2bl.Dcomp)]), xlabel=r'$V$', ylabel=r'$Pxz$', fname='0A-FFvsV-D={0}'.format(f2al.Dcomp), names=['f2a-ftwall'], log=(True, True))

    PvS_r[:, 1, :] = PvS_r[:, 1, :] * Axy
    PvS_l[:, 1, :] = PvS_l[:, 1, :] * Axy
    PvS_r[:,3,:] = PvS_r[:,3,:] * Axy
    PvS_l[:,3,:] = PvS_l[:,3,:] * Axy


    myErrorPlot([abs(PvS_r), abs(PvS_l)], llabels=(2, ['2a-N={0}-M={1}-D={2}'.format(f2al.N,f2al.M,f2al.Dcomp), '2b-N={0}-M={1}-D={2}'.format(f2bl.N,f2bl.M,f2bl.Dcomp)]), xlabel=r'$V$', ylabel=r'$Pxz$',fname='0A-FFvsV-D={0}'.format(f2al.Dcomp), names=['f2a-ftwallAxy'], log=(True, True))
    '''

    #Single
    r'''

    # Equilibrium
    DPSra = f2aR.profiles(['bbdpe'],  Vis=[0], Z2D=False)
    DPSla = f2al.profiles(['bbdpe'],  Vis=[0], Z2D=False)
    DPSrb = f2bR.profiles(['bbdpe'],  Vis=[0], Z2D=False)
    DPSlb = f2bl.profiles(['bbdpe'],  Vis=[0], Z2D=False)
    DPSr = np.hstack((DPSra,DPSrb))
    DPSl = np.hstack((DPSla,DPSlb))
    myPlot([DPSr, DPSl], llabels = (1,['2a-N={0}'.format,'Linear']), xlim=[0, 50], xlabel='$z$', ylabel=r'$\phi(z)$',fname='1A-EquilProfiles', names=['f2a'])

    # Compression
    DPSra = f2aR.profiles(['bbdpc', 'abdpc', 'tbdpc'],  Vis=[0], Z2D=True)
    DPSla = f2al.profiles(['bbdpc', 'abdpc', 'tbdpc'], Vis=[0], Z2D=True)
    DPSrb = f2bR.profiles(['bbdpc', 'abdpc', 'tbdpc'],  Vis=[0], Z2D=True)
    DPSlb = f2bl.profiles(['bbdpc', 'abdpc', 'tbdpc'], Vis=[0], Z2D=True)
    DPSr = np.hstack((DPSra,DPSrb))
    DPSl = np.hstack((DPSla,DPSlb))
    myPlot([DPSr, DPSl], llabels=(0, ['Ring', 'Linear']), xlim=[0, 1], xlabel='$z/D$', ylabel=r'$\phi(z)$',fname='2A-CompProfiles', names=['f2a'])

    # Shearing

    Vs = [0, 2]

    for V in Vs:
        rdata = f2aR.profiles(['temps', 'abdps', 'velps'], Vis=[V],Z2D=True)
        ldata = f2al.profiles(['temps', 'abdps', 'velps'], Vis=[V],Z2D=True)
        myPlot([rdata, ldata], llabels=(2, ['Temperature', 'Density', 'Velocity']), axis2=[0, 0, 1],xlim=[0, 1], xlabel='$z/D$', ylabel=r'$\phi(z), T$', y2label='V', fname='3A-TDVProfiles',names=['{0}'.format(V)])

    RvS_ra = f2aR.thermo(3, ['v_srate_m', 'v_aveRgx'], Vis=0, avg=True)
    RvS_la = f2al.thermo(3, ['v_srate_m', 'v_aveRgx'], Vis=0, avg=True)
    RvS_rb = f2bR.thermo(3, ['v_srate_m', 'v_aveRgx'], Vis=0, avg=True)
    RvS_lb = f2bl.thermo(3, ['v_srate_m', 'v_aveRgx'], Vis=0, avg=True)
    R2vS_ra = Rg2x(RvS_ra)
    R2vS_la = Rg2x(RvS_la)
    R2vS_rb = Rg2x(RvS_rb)
    R2vS_lb = Rg2x(RvS_lb)
    R2vS_r = np.hstack((R2vS_ra,R2vS_rb))
    R2vS_l = np.hstack((R2vS_la,R2vS_lb))
    myErrorPlot([R2vS_r, R2vS_l], llabels=(0, ['Ring', 'Linear']), xlabel=r'$\.\gamma$', ylabel=r'$Rgx$', fname='3B-Rg2xvsSrate',names=['f2a'], log=(True, True))

    src_ra = srate_crit(R2vS_ra)
    src_la = srate_crit(R2vS_la)
    src_rb = srate_crit(R2vS_rb)
    src_lb = srate_crit(R2vS_lb)

    R2vW_ra = Weissenberg(R2vS_ra, src_ra)
    R2vW_la = Weissenberg(R2vS_la, src_la)
    R2vW_rb = Weissenberg(R2vS_rb, src_rb)
    R2vW_lb = Weissenberg(R2vS_lb, src_lb)
    R2vW_r = np.hstack((R2vW_ra,R2vW_rb))
    R2vW_l = np.hstack((R2vW_la,R2vW_lb))

    myErrorPlot([R2vW_r, R2vW_l], llabels=(0, ['Ring', 'Linear']), xlabel=r'$W$', ylabel=r'$Rgx$', fname='3C-Rg2xvsW',names=['f2a'], log=(True, True))

    PvS_ra = f2aR.thermo(3, ['v_srate_m', 'pxz', 'pzz'],  Vis=0, avg=True)
    PvS_la = f2al.thermo(3, ['v_srate_m', 'pxz', 'pzz'],  Vis=0, avg=True)
    PvS_rb = f2bR.thermo(3, ['v_srate_m', 'pxz', 'pzz'],  Vis=0, avg=True)
    PvS_lb = f2bl.thermo(3, ['v_srate_m', 'pxz', 'pzz'],  Vis=0, avg=True)
    PvS_r = np.hstack((PvS_ra,PvS_rb))
    PvS_l = np.hstack((PvS_la,PvS_lb))

    myErrorPlot([PvS_r, PvS_l], llabels=(0, ['Ring', 'Linear']), xlabel=r'$\.\gamma$', ylabel=r'$Pxz$', fname='3C-PvsSrate', names=['f2a'], log=(True, True))

    FvS_ra = COF(PvS_ra)
    FvS_la = COF(PvS_la)
    FvS_rb = COF(PvS_rb)
    FvS_lb = COF(PvS_lb)
    FvS_r = np.hstack((FvS_ra,FvS_rb))
    FvS_l = np.hstack((FvS_la,FvS_lb))

    myErrorPlot([FvS_r, FvS_l], llabels=(0, ['Ring', 'Linear']), xlabel=r'$\.\gamma$', ylabel=r'$\mu$', fname='3D-CoFvsSrate',names=['f2a'], log=(True, True))

    FvW_ra = Weissenberg(FvS_ra, src_ra)
    FvW_la = Weissenberg(FvS_la, src_ra)
    FvW_rb = Weissenberg(FvS_rb, src_rb)
    FvW_lb = Weissenberg(FvS_lb, src_rb)
    FvW_r = np.hstack((FvW_ra,FvW_rb))
    FvW_l = np.hstack((FvW_la,FvW_lb))
    myErrorPlot([FvW_r, FvW_l], llabels=(0, ['Ring', 'Linear']), xlabel=r'$W$', ylabel=r'$\mu$', fname='3D-CoFvsW',names=['f2a'], log=(True, True))
    '''
    t1 = time.perf_counter()
    print("Time Elapsed: {0:.2f} s".format(t1-t0))
