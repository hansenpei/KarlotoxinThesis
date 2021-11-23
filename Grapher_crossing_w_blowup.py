# -*- coding: utf-8 -*-
"""
Graphing file for using real s(c(x)) with crossing, with blow up at the sink

@author: Hansen
"""

import os
os.chdir('D:\Research with Dr Rossi\Python Scratch')

import numpy as np
import matplotlib.pyplot as plt
from OOP_theoretical_value import s_module
from typing import Callable
import scipy.optimize
from scipy.interpolate import interp1d
from FiniteDiffv2 import UpwindLFNoFlux
import time

#%% Construct alternative s_LF for faster computation
gamma0 = 0.532154
s_mod = s_module(gamma=gamma0, a=3, b=16)
s_original = s_mod.speed

# Polynomial approximation of s_original(x) for faster calculation

xdata_l = np.linspace(0,gamma0,10000, endpoint=False)
xdata_r = np.linspace(gamma0,1,10000, endpoint=True)
xdata = np.concatenate((xdata_l, xdata_r))

s_polyinterp = interp1d(xdata, s_original(xdata), kind = 'linear')  # This function is faster

s_coeff_l = np.polyfit(xdata_l, s_original(xdata_l), deg = 4)
s_coeff_r = np.polyfit(xdata_r, s_original(xdata_r), deg = 4)

s_polyfit = lambda x: np.poly1d(s_coeff_l)(x)*(x<gamma0) + np.poly1d(s_coeff_r)(x)*(x>=gamma0)

# Find x1, x3
x1 = scipy.optimize.bisect(lambda x: s_polyfit(x) -1, 0.0, 0.5)
x3 = scipy.optimize.bisect(lambda x: s_polyfit(x) -1, 0.5, 1)

def derv_sink(s: Callable, x1: float) -> float:
    """
    Calculate the numerical derivative of s'(x_1)
    """
    num_derv = (s(x1 + 1e-4) - s(x1 - 1e-4)) / (2e-4)

    return num_derv

print(derv_sink(s_polyinterp, x1))

#%% plotting to check polyfit of s
plt.figure(dpi = 300)

plt.plot(xdata_l, s_original(xdata_l), color = 'blue', label = r'original $s$')
plt.plot(xdata_r, s_original(xdata_r), color = 'blue')
plt.plot(xdata_l, s_polyinterp(xdata_l), linestyle = '--', color = 'red', label = r'polyfit $s$')
plt.plot(xdata_r, s_polyinterp(xdata_r), linestyle = '--', color = 'red')
plt.legend(loc = 'best')
plt.grid()
plt.tight_layout()
plt.show()

#%% instantiate and run the solver
my_UW = UpwindLFNoFlux(s = s_polyinterp,
                   gamma = None,
                   x1 = x1,
                   x3 = x3,
                   N = 800,
                   CFL = 0.4,
                   T = 10)
my_UW.run(if_plot=False)
#%% Data Handling
x_num = my_UW.x_vec
Pp_num = my_UW.Pp[:,-1]
Pm_num = my_UW.Pm[:,-1]


#%% graphing for Pp, Pm, s
import matplotlib.lines as mlines
Pp_marker = mlines.Line2D([], [], color='darkgoldenrod', marker='X', linestyle='-',
                          linewidth=3, markersize=8, label=r'$P+$')
Pm_marker = mlines.Line2D([], [], color='blueviolet', marker='d', linestyle='-',
                          linewidth=3, markersize=8, label=r'$P-$')
Sig_marker = mlines.Line2D([], [], color='brown', marker='o', linestyle='-',
                          linewidth=3, markersize=8, label=r'$\Sigma$')
Dlt_marker = mlines.Line2D([], [], color='royalblue', marker='^', linestyle='-',
                          linewidth=3, markersize=8, label=r'$\Delta$')
xi_marker = mlines.Line2D([], [], color='darkorange', marker='v', linestyle='-',
                          linewidth=3, markersize=8, label=r'$\xi$')
eta_marker = mlines.Line2D([], [], color='forestgreen', marker='o', linestyle='-',
                          linewidth=3, markersize=8, label=r'$\eta$')
sm1_marker = mlines.Line2D([], [], color='black', linestyle='--',
                          linewidth=3, markersize=5, label=r'$s-1$')
c_marker = mlines.Line2D([], [], linestyle='-.',
                          linewidth=3, markersize=5, label=r'$c$')
plt.style.use('seaborn-paper')

fig2 = plt.figure(figsize = (13,8), dpi = 200)
step_size = 2 # controls plot density
# plot finite difference method result
plt.plot(x_num[::step_size], Pp_num[::step_size],'X', markersize=8, label = '$P+$', color = 'darkgoldenrod')
plt.plot(x_num[::step_size], Pm_num[::step_size],'d', markersize=8, label = '$P-$', color = 'blueviolet')
#plt.plot(x_num[::step_size], xi_num[::step_size] ,'v', markersize=8, label = '$\\xi$', color = 'darkorange')
step_size_1 = 1
plt.plot(x_num[::step_size_1], Pp_num[::step_size_1],'-', linewidth=3, label = '$P+$', color = 'darkgoldenrod')
plt.plot(x_num[::step_size_1], Pm_num[::step_size_1],'-', linewidth=3, label = '$P-$', color = 'blueviolet')
#plt.plot(x_num[::step_size_1], xi_num[::step_size_1] ,'-', linewidth=3, label = '$\\xi$', color = 'darkorange')
# plot analytical result
# plt.plot(x_aly, Pp_aly,'-', linewidth=3, color = 'darkgoldenrod')
# plt.plot(x_aly, Pm_aly,'-', linewidth=3, color = 'blueviolet')
# #plt.plot(x_aly, [-1]*len(x_aly) ,'-', linewidth=3, color = 'forestgreen')
# plt.plot(x_aly, xi_aly  ,'-', linewidth=3, color = 'darkorange')
plt.plot(xdata, s_polyinterp(xdata) - 1, '--', linewidth=3, label = '$s - 1$',color = 'black')
#plt.plot(x_aly, c_aly ,'-.', linewidth=3, label = '$c$')

plt.legend(bbox_to_anchor=(0.82,0.7), handles=[Pp_marker, Pm_marker,
                                               sm1_marker],
           loc="center left", borderaxespad=0, fontsize=20)
plt.xticks(fontsize=25)   # fontsize of the tick labels
plt.yticks(fontsize=25)    # fontsize of the tick labels
plt.xlabel('x', fontsize = 25)
plt.ylabel('Value', fontsize= 25)

plt.grid()
plt.xlim(0,1)
plt.ylim(-0.5,3.7)
plt.xticks([0,0.25,0.5,0.75,1])
plt.yticks([0,1,2,3])

# Vertical line at the sink and source
plt.vlines(0.19884821244195336, -1.2, 4, linestyles='dotted', colors='grey', linewidth=3)  # At the sink
plt.vlines(0.6383641464795798, -1.2, 4, linestyles='dotted', colors='grey', linewidth=3)  # At the source
plt.gcf().text(0.25, 0.07, r'$x_1$', fontsize=25)
plt.gcf().text(0.6, 0.07, r'$x_3$', fontsize=25)

# annotate x1 and x3 in arrows
# plt.annotate('$x_1$', xy=(0.19884821244195336 - 0.01, 0 - 0.05),  xycoords='data',
#             xytext=(0.15, -0.5), textcoords='data', fontsize = 20,
#             arrowprops=dict(facecolor='black', shrink=0.02),
#             horizontalalignment='right', verticalalignment='top',
#             )

# plt.annotate('$x_3$', xy=(0.6383641464795798 + 0.01, 0 - 0.05),  xycoords='data',
#             xytext=(0.72, -0.5), textcoords='data', fontsize = 20,
#             arrowprops=dict(facecolor='black', shrink=0.02),
#             horizontalalignment='right', verticalalignment='top',
#             )
plt.show()