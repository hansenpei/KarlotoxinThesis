# -*- coding: utf-8 -*-
"""
Controlled experiment where the chars can be calculated exactly.

@author: Hansen
"""

import os
os.chdir('D:\Research with Dr Rossi\Python Scratch')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from MC_stationary_toxin_cos import MC_Stationary_Toxin_Cos
from MC_linear_speed_roots import MC_Linear_Speed_Roots

#%% Initiate instances, low and high num of agents
MCs_10000 = MC_Stationary_Toxin_Cos(Ntimenodes = 200, endtime=10, iterations = 10000)
MCs_100000 = MC_Stationary_Toxin_Cos(Ntimenodes = 200, endtime=10, iterations = 100000)
MCl_10000 = MC_Linear_Speed_Roots(Ntimenodes = 200, endtime=10, iterations = 10000)
MCl_100000 = MC_Linear_Speed_Roots(Ntimenodes = 200, endtime=10, iterations = 100000)

bins = np.linspace(0, 1, 101, endpoint = True)
x_data = bins[:-1]+1/2*(bins[1]-bins[0])
#%% Run instance 1
MCs_10000.run_all_agents()

# Record the final density
Sigma_num_s_10000, _ = np.histogram(MCs_10000.locmatrix[:,200], bins = bins, density=True)
del MCs_10000
#%% Run instance 2
MCs_100000.run_all_agents()

# Record the final density
Sigma_num_s_100000, _ = np.histogram(MCs_100000.locmatrix[:,200], bins = bins, density=True)
del MCs_100000
#%% Run instance 3
MCl_10000.run_all_agents()

# Record the final density
Sigma_num_l_10000, _ = np.histogram(MCl_10000.locmatrix[:,200], bins = bins, density=True)
del MCl_10000
#%% Run instance 4
MCl_100000.run_all_agents()

# Record the final density
Sigma_num_l_100000, _ = np.histogram(MCl_100000.locmatrix[:,200], bins = bins, density=True)
del MCl_100000
#%% Import analytical sol for both cases
# For the lin speed case
data_l = pd.read_csv('D:\\Research with Dr Rossi\\Mathematica Scratch\\analytical_w_crossing_logplus18.csv')
x_aly_l = data_l['xdata'].to_numpy()
xi_aly_l = data_l['xidata'].to_numpy()
s_aly_l = data_l['sdata'].to_numpy()
eta_aly_l = data_l['etadata'].to_numpy()
Pp_aly_l = (xi_aly_l + eta_aly_l) / (2 * (s_aly_l - 1))
Pm_aly_l = (xi_aly_l - eta_aly_l) / (2 * (s_aly_l + 1))
Sigma_aly_l = Pp_aly_l + Pm_aly_l
Delta_aly_l = Pp_aly_l - Pm_aly_l

# For the stationary case
data_s = pd.read_csv(r'D:\Research with Dr Rossi\Mathematica Scratch\Fourier Series Sol Stationary Toxin Cosine.csv')
x_aly_s = data_s[r'xdata'].to_numpy()
Sigma_aly_s = data_s[r'Sig10'].to_numpy()


#%% Plot analytical data against MC data

#subplots: beginning, middle 1, middle 2, end
plt.style.use('seaborn-paper')
linewidth = 4
linestyle = '--'
bar_width = x_data[1]-x_data[0]
#plt.rcParams.update({'font.size': 7})
fig, ax = plt.subplots(2,2, figsize=(20, 12), dpi = 100)
all_ax = [ax[0,0], ax[0,1], ax[1,0], ax[1,1]]
ax[0, 0].bar(x_data, Sigma_num_s_10000, color='steelblue', alpha=0.6, width = bar_width,
              label = 'Simulated $\Sigma$')
ax[0,0].plot(x_aly_s, Sigma_aly_s, color = 'black', linestyle = linestyle, linewidth=linewidth,
              label = 'Analytical $\Sigma$')
ax[0,0].set_ylim(0,3)
#ax[0,0].set_title('t=0', fontsize=30)
fig.tight_layout(pad=7)
xtick = [0,0.5,1]
ytick = [0,1,2,3,4]

ax[1,0].bar(x_data, Sigma_num_l_10000, color='steelblue', alpha=0.6, width = bar_width)
ax[1,0].plot(x_aly_l, Sigma_aly_l, color = 'black', linestyle = linestyle, linewidth=linewidth)
ax[1,0].set_ylim(0,3)
#ax[1,0].set_title('t=0.5', fontsize=30)

ax[0,1].bar(x_data, Sigma_num_s_100000, color='steelblue', alpha=0.6, width = bar_width)
ax[0,1].plot(x_aly_s, Sigma_aly_s, color = 'black', linestyle = linestyle, linewidth=linewidth)
ax[0,1].set_ylim(0,3)
#ax[0,1].set_title('t=0.15', fontsize=30)

ax[1, 1].bar(x_data, Sigma_num_l_100000, color='steelblue', alpha=0.6, width = bar_width)
ax[1,1].plot(x_aly_l, Sigma_aly_l, color = 'black', linestyle = linestyle, linewidth=linewidth)
ax[1,1].set_ylim(0,3)
#ax[1,1].set_title('t=10', fontsize=30)

# s info
fig.text(0.27, 0.85, '$s_+$ Has No Roots', va='center', fontsize=30)
fig.text(0.75, 0.85, '$s_+$ Has No Roots', va='center', fontsize=30)
fig.text(0.3, 0.4, '$s_+$ Has Roots', va='center', fontsize=30)
fig.text(0.75, 0.4, '$s_+$ Has Roots', va='center', fontsize=30)

# Num Agents
fig.text(0.27, 0.8, 'N=10000', va='center', fontsize=30)
fig.text(0.75, 0.8, 'N=100000', va='center', fontsize=30)
fig.text(0.3, 0.35, 'N=10000', va='center', fontsize=30)
fig.text(0.75, 0.35, 'N=100000', va='center', fontsize=30)

# 'x' label
fig.text(0.25, 0.5, '$x$', va='center', fontsize=30)
fig.text(0.73, 0.5, '$x$', va='center', fontsize=30)
fig.text(0.25, 0.03, '$x$', va='center', fontsize=30)
fig.text(0.73, 0.03, '$x$', va='center', fontsize=30)

# 'Sigma' label
fig.text(0.02, 0.75, r'$\Sigma$', va='center', fontsize=30)
fig.text(0.50, 0.75, r'$\Sigma$', va='center', fontsize=30)
fig.text(0.02, 0.30, r'$\Sigma$', va='center', fontsize=30)
fig.text(0.50, 0.30, r'$\Sigma$', va='center', fontsize=30)

# ABCD label
fig.text(0.08, 0.88, r'$\it{(a)}$', va='center', fontsize=40)
fig.text(0.56, 0.88, r'$\it{(b)}$', va='center', fontsize=40)
fig.text(0.08, 0.42, r'$\it{(c)}$', va='center', fontsize=40)
fig.text(0.56, 0.42, r'$\it{(d)}$', va='center', fontsize=40)

# fig.text(0.02, 0.5, 'Density', va='center', rotation='vertical', fontsize=30)
# fig.text(0.45, 0.02, 'Location', va='center', fontsize=30)
fig.legend(bbox_to_anchor=(0.08,0.85), loc="upper left", borderaxespad=0, fontsize=25)

for each_ax in all_ax:
    each_ax.tick_params(axis='both', which='major', labelsize=28)
    each_ax.set_xticks(xtick)
    each_ax.set_yticks(ytick)
fig.show()


