# -*- coding: utf-8 -*-
"""
Use different fin diff methods to verify the analytical solution
Post-defense

@author: Hansen
"""
import os
os.chdir('D:\Research with Dr Rossi\Python Scratch')

import numpy as np
import matplotlib.pyplot as plt
from FiniteDiffv2 import FiniteDiff, RichtmyerLaxWendroff, UpwindLF, UpwindLFHalfSink, UpwindLFLF
from FiniteDiffv2 import UpwindLFNoFlux, UpwindLFNoFluxItp
import time
import pickle
import pandas as pd

#%% Import and format analytical sol
os.chdir("D:\\Research with Dr Rossi\\Mathematica Scratch")
data = pd.read_csv('analytical_w_crossing_logplus18.csv')
x_aly = data['xdata'].to_numpy()
xi_aly = data['xidata'].to_numpy()
s_aly = data['sdata'].to_numpy()
eta_aly = data['etadata'].to_numpy()

# Analytical Solution Data Processing
Pp_aly = (xi_aly + eta_aly) / (2 * (s_aly - 1))
Pm_aly = (xi_aly - eta_aly) / (2 * (s_aly + 1))
Sigma_aly = Pp_aly + Pm_aly
Delta_aly = Pp_aly - Pm_aly

#%% Old upwind run
x1 = 3/8
x3 = 5/8
s0 = 1
def s(x):
    speed = (-1*(x-x1)+1) * (x<=1/2) \
           + (1*(x-x3)+1) * (x>1/2)
    return speed

#% UW run
my_UW1 = UpwindLF(s = s,
                x1 = x1,
                gamma = None,
                x3 = x3,
                N = 201,
                CFL = 0.4,
                F = 1,
                T = 10)
start_time = time.time()

my_UW1.run()

print("--- {:.1f}s seconds ---".format(time.time() - start_time))

#%% Plot against analytical sol at steady state
Sigma_UW1 = my_UW1.Pp + my_UW1.Pm
Delta_UW1 = my_UW1.Pp - my_UW1.Pm

plt.style.use('seaborn-paper')
linewidth = 2
linestyle = '-'
fig, ax = plt.subplots(1,1, figsize=(10, 6), dpi = 190)
# Numerical
ax.plot(my_UW1.x_vec, Sigma_UW1[:, -1], color = 'steelblue', linestyle = linestyle, linewidth=linewidth,
        label = 'Modified Upwind $\Sigma$')
# Analytical
ax.plot(x_aly, Sigma_aly, color = 'black', linestyle = '--', linewidth=linewidth,
        label = 'Analytical $\Sigma$')
ax.set_ylim(0,5)
ax.grid()
ax.legend()
plt.show()


#%% Coeff 1/2 at sink upwind run
x1 = 3/8
x3 = 5/8
s0 = 1
def s(x):
    speed = (-1*(x-x1)+1) * (x<=1/2) \
           + (1*(x-x3)+1) * (x>1/2)
    return speed

#% UW run
my_UW2 = UpwindLFHalfSink(
    s = s,
    x1 = x1,
    gamma = None,
    x3 = x3,
    N = 201,
    CFL = 0.4,
    F = 1,
    T = 10)
start_time = time.time()

my_UW2.run()

print("--- {:.1f}s seconds ---".format(time.time() - start_time))

#%% Plot against analytical sol at steady state
Sigma_UW2 = my_UW2.Pp + my_UW2.Pm
Delta_UW2 = my_UW2.Pp - my_UW2.Pm

plt.style.use('seaborn-paper')
linewidth = 2
linestyle = '-'
fig, ax = plt.subplots(1,1, figsize=(10, 6), dpi = 190)
# Numerical
ax.plot(my_UW2.x_vec, Sigma_UW2[:, -1], color = 'steelblue', linestyle = linestyle, linewidth=linewidth,
        label = 'Modified Upwind Half $\Sigma$')
# Analytical
ax.plot(x_aly, Sigma_aly, color = 'black', linestyle = '--', linewidth=linewidth,
        label = 'Analytical $\Sigma$')
ax.set_ylim(0,5)
ax.grid()
ax.legend()
plt.show()

#%% Coeff 1/2 at sink upwind run
x1 = 3/8
x3 = 5/8
s0 = 1
def s(x):
    speed = (-1*(x-x1)+1) * (x<=1/2) \
           + (1*(x-x3)+1) * (x>1/2)
    return speed

#% UW run
my_UW3 = UpwindLFLF(
    s = s,
    x1 = x1,
    gamma = None,
    x3 = x3,
    N = 401,
    CFL = 0.4,
    F = 1,
    T = 10)
start_time = time.time()

my_UW3.run()

print("--- {:.1f}s seconds ---".format(time.time() - start_time))

#%% Plot against analytical sol at steady state
Sigma_UW3 = my_UW3.Pp + my_UW3.Pm
Delta_UW3 = my_UW3.Pp - my_UW3.Pm

plt.style.use('seaborn-paper')
linewidth = 2
linestyle = '-'
fig, ax = plt.subplots(1,1, figsize=(10, 6), dpi = 190)
# Numerical
ax.plot(my_UW3.x_vec, Sigma_UW3[:, -1], color = 'steelblue', linestyle = linestyle, linewidth=linewidth,
        label = 'Modified Upwind LFLF $\Sigma$')
# Analytical
ax.plot(x_aly, Sigma_aly, color = 'black', linestyle = '--', linewidth=linewidth,
        label = 'Analytical $\Sigma$')
ax.set_ylim(0,5)
ax.grid()
ax.legend()
plt.show()

#%% No flux at sink upwind run
x1 = 3/8
x3 = 5/8
s0 = 1
def s(x):
    speed = (-1*(x-x1)+1) * (x<=1/2) \
           + (1*(x-x3)+1) * (x>1/2)
    return speed

#% UW run
my_UW4 = UpwindLFNoFlux(
    s = s,
    x1 = x1,
    gamma = None,
    x3 = x3,
    N = 1001,
    CFL = 0.4,
    F = 1,
    T = 10)
start_time = time.time()

my_UW4.run()

print("--- {:.1f}s seconds ---".format(time.time() - start_time))

#%% Plot against analytical sol at steady state
Sigma_UW4 = my_UW4.Pp + my_UW4.Pm
Delta_UW4 = my_UW4.Pp - my_UW4.Pm

plt.style.use('seaborn-paper')
linewidth = 2
linestyle = '-'
fig, ax = plt.subplots(1,1, figsize=(10, 6), dpi = 190)
# Numerical
ax.plot(my_UW4.x_vec, Sigma_UW4[:, -1], color = 'steelblue', linestyle = linestyle, linewidth=linewidth,
        label = 'Modified Upwind LFNoFlux $\Sigma$')
# Analytical
ax.plot(x_aly, Sigma_aly, color = 'black', linestyle = '--', linewidth=linewidth,
        label = 'Analytical $\Sigma$')
ax.set_ylim(0,5)
ax.grid()
ax.legend()
plt.show()

#%% Convergence study of No flux FD scheme
Ncells = [50, 100, 200, 400, 800, 1600]
Pps = {}
xs = {}
etas = {}

x1 = 3/8
x3 = 5/8
s0 = 1
def s(x):
    speed = (-1*(x-x1)+1) * (x<=1/2) \
           + (1*(x-x3)+1) * (x>1/2)
    return speed

start_time = time.time()
for N in Ncells:
    my_UW = UpwindLFNoFlux(s = s,
                    x1 = x1,
                    gamma = None,
                    x3 = x3,
                    N = N,
                    CFL = 0.4,
                    T = 10)
    my_UW.run(if_plot=False)
    Pps[N] = my_UW.Pp[:,-1]
    xs[N] = my_UW.x_vec
    etas[N] = my_UW.eta[:,-1]
print("--- {:.1f}s seconds ---".format(time.time() - start_time))

#%% save data
os.chdir('D:\Research with Dr Rossi\Python Scratch')
with open('etas_pls18_Convergence_noflux.pickle', 'wb') as handle:
    pickle.dump(etas, handle, protocol=pickle.HIGHEST_PROTOCOL)

#%% Convergence check in Sigma
# Create 1D interp function for Pp_aly()
from scipy.interpolate import interp1d

Pp_aly_itp = interp1d(x_aly, Pp_aly, kind = 'linear')


# Check and plot
import matplotlib.lines as mlines
error_marker = mlines.Line2D([], [], color='royalblue', marker='s', linestyle='--',
                             linewidth=3, markersize=10, label=r'Averaged Error in $L^1$ for $\Sigma$')
Pm_marker = mlines.Line2D([], [], color='lightcoral', linestyle='-.',
                          linewidth=3, markersize=8, label=r'Slope $=-1$')

Ncells = list(Pps.keys())

single_Pp_diff = [np.mean(np.abs(Pps[ncell] - Pp_aly_itp(xs[ncell]))) for ncell in Ncells]
single_Pp_diff_tick = [np.format_float_scientific(num, precision = 2)
                        for num in single_Pp_diff]
plt.style.use('seaborn-paper')
plt.figure(dpi = 200, figsize=(13,5))
plt.plot(Ncells, single_Pp_diff, '--s', linewidth=3, markersize=10, label='Error', color='royalblue')
plt.plot((50, 800), (2.66e-2, 1.68e-3), linestyle='-.', linewidth=3, color='lightcoral')
plt.xscale('log')
plt.yscale('log')
plt.gcf().text(0.5, 0.01, '$N$', rotation='horizontal', fontsize=25)
plt.gcf().text(0.01, 0.4, r'Value', rotation='vertical', fontsize=25)
plt.gcf().text(0.15, 0.18, r'$\it{A}$', rotation='horizontal', fontsize=25)
plt.grid()
plt.legend(bbox_to_anchor=(0.52,0.8), handles=[error_marker, Pm_marker],
           loc="center left", borderaxespad=0, fontsize=20)
plt.tight_layout(rect=(0.08, 0.05, 0.95, 0.95))
plt.xticks(ticks=Ncells, labels=Ncells, fontsize=20)
plt.yticks(ticks=single_Pp_diff, labels=single_Pp_diff_tick, fontsize=15)
plt.show()

#%% Convergence check in eta
import matplotlib.lines as mlines
error_marker = mlines.Line2D([], [], color='royalblue', marker='s', linestyle='--',
                             linewidth=3, markersize=10, label=r'Averaged Error in $L^1$ for $\eta$')
Pm_marker = mlines.Line2D([], [], color='lightcoral', linestyle='-.',
                          linewidth=3, markersize=8, label=r'Slope $=-1$')

Ncells = list(etas.keys())

eta_theo = -1.015123552349955 #-0.9838459177298593
single_eta_diff =  [np.mean(np.abs(etas[ncell] - eta_theo)) for ncell in Ncells]
single_eta_diff_tick = [np.format_float_scientific(num, precision = 2)
                        for num in single_eta_diff]
plt.style.use('seaborn-paper')
plt.figure(dpi = 200, figsize=(13,5))
plt.plot(Ncells, single_eta_diff, '--s', linewidth=3, markersize=10, label='Error', color='royalblue')
plt.plot((50, 800), (2.66e-2, 1.68e-3), linestyle='-.', linewidth=3, color='lightcoral')
plt.xscale('log')
plt.yscale('log')
plt.gcf().text(0.5, 0.01, '$N$', rotation='horizontal', fontsize=25)
plt.gcf().text(0.01, 0.4, r'Value', rotation='vertical', fontsize=25)
plt.gcf().text(0.15, 0.18, r'$\it{A}$', rotation='horizontal', fontsize=25)
plt.grid()
plt.legend(bbox_to_anchor=(0.52,0.8), handles=[error_marker, Pm_marker],
           loc="center left", borderaxespad=0, fontsize=20)
plt.tight_layout(rect=(0.08, 0.05, 0.95, 0.95))
plt.xticks(ticks=Ncells, labels=Ncells, fontsize=20)
plt.yticks(ticks=single_eta_diff, labels=single_eta_diff_tick, fontsize=15)
plt.show()

#%% No flux at sink (interpolated) upwind run
x1 = 3/8
x3 = 5/8
s0 = 1
def s(x):
    speed = (-1*(x-x1)+1) * (x<=1/2) \
           + (1*(x-x3)+1) * (x>1/2)
    return speed

#% UW run
my_UW5 = UpwindLFNoFluxItp(
    s = s,
    x1 = x1,
    gamma = None,
    x3 = x3,
    N = 201,
    CFL = 0.4,
    F = 1,
    T = 10)
start_time = time.time()

my_UW5.run()

print("--- {:.1f}s seconds ---".format(time.time() - start_time))

#%% Plot against analytical sol at steady state
Sigma_UW5 = my_UW5.Pp + my_UW5.Pm
Delta_UW5 = my_UW5.Pp - my_UW5.Pm

plt.style.use('seaborn-paper')
linewidth = 2
linestyle = '-'
fig, ax = plt.subplots(1,1, figsize=(10, 6), dpi = 190)
# Numerical
ax.plot(my_UW5.x_vec, Sigma_UW5[:, -1], color = 'steelblue', linestyle = linestyle, linewidth=linewidth,
        label = 'Modified Upwind LFNoFlux $\Sigma$')
# Analytical
ax.plot(x_aly, Sigma_aly, color = 'black', linestyle = '--', linewidth=linewidth,
        label = 'Analytical $\Sigma$')
ax.set_ylim(0,5)
ax.grid()
ax.legend()
plt.show()
