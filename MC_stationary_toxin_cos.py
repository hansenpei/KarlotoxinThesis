# -*- coding: utf-8 -*-
"""
Agent based 1D simulator, stationary toxin

@author: Hansen
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d
import time

class MC_Stationary_Toxin_Cos:
    def __init__(self,
                 s = None,
                 F = 1,
                 iterations = 3000, # num of agents
                 endtime = 20, # when the simulation ends
                 domainsize = 1, # x from 0 to domainsize
                 Ntimenodes = 200 # stores location and direction info at those nodes, not including t = 0
                 ):
        # MC parameters
        self.iterations = iterations
        self.endtime = endtime
        self.epsilon = 1E-3 # Param for expanding x on the right end for calculating y
        # Model parameters
        self.domainsize = domainsize
        self.s = self._setup_s(s)
        self.F = F
        self._setup_ylu() #upperbound, lowerbound in y domain
        self.x_from_y, self.y_from_x = self._build_y_from_x_interp() # interpolated for faster calc
        # Plotting and analysis variables
        self._setup_time_evolution_plot_vars(Ntimenodes)

    def run_all_agents(self, if_plot = True):
        """
        Run all the agents and store info in locmatrix and dirmatrix
        """
        milestone = 0 # for printing progress
        display_nodes = [self.iterations*i//100 for i in range(1,100,5)] + [self.iterations]

        start_time = time.time()

        # run through the iteration
        for agent_idx in range(self.iterations):
            agent_loc_history, agent_dir_history = self.run_one_agent(agent_idx)
            self.locmatrix[agent_idx][:] = agent_loc_history
            self.dirmatrix[agent_idx][:] = agent_dir_history

            # print progress
            if (agent_idx == display_nodes[milestone]):
                print("{:.1f} percent finished".format(agent_idx/self.iterations *100))
                milestone += 1

        print("--- {:.1f}s seconds ---".format(time.time() - start_time))

#    @numba.njit(parallel=True, debug=True)
    def run_one_agent(self, agent_idx: int = 0):
        """
        Run the simulation for one agent with index agent_idx.
        Stores location and idrection info at given time nodes
        """
        schedule, directions = self._generate_schedule_and_directions()
        loc = np.zeros_like(schedule, dtype=np.float64) # corresponding loc regarding schedule

        ## start simulation ##
        # set up initial location
        loc[0] = np.random.rand()*self.domainsize

        # compute location at all the time nodes from the master schedule
        for cur_step in range(1, len(schedule)):
            # note that the current location needs info from the previous step
            loc[cur_step] = self._trueloc(direction = directions[cur_step-1],
                                          location = loc[cur_step-1],
                                          traveltime = schedule[cur_step] - schedule[cur_step-1])

        # only take value from the sample time nodes
        sample_nodes_pointer = 0
        sample_loc = np.zeros_like(self.timenodes, dtype=np.float32)
        sample_dir = np.zeros_like(self.timenodes, dtype=np.int32)
        for index, value in enumerate(loc):
            if schedule[index] == self.timenodes[sample_nodes_pointer]:
                sample_loc[sample_nodes_pointer] = value
                sample_dir[sample_nodes_pointer] = directions[sample_nodes_pointer]
                sample_nodes_pointer += 1

        return sample_loc, sample_dir

    def hist_plot(self, node_number, bins = 10):
        """
        Plot the location histogram at a given node number from 0 to self.num_of_time_samples
        """
        fig = plt.figure(dpi = 300)
        ax = fig.add_subplot(111)
        ax.hist(self.locmatrix[:,node_number], density= True, bins = bins)

        # match with inverse of s(x)
        xdata = np.linspace(0,self.domainsize, 1001)
        inv_s_data = list(map(lambda x: 1/self.s(x)/self.yupper, xdata))
        ax.plot(xdata, inv_s_data, label = r'$1/s(x)$')

        ax.set_title('Node = {}, Time = {:.1f}'.format(node_number, node_number/self.num_of_time_samples*self.endtime))
        fig.tight_layout()

        fig.show()

    def _generate_schedule_and_directions(self):
        """
        schedule: at those times we are interested in the agent's location and direction
            size: self.num_of_time_samples + 1
            source: sampling time nodes merged with flip schedule, cut off at self.endtime

        directions: right of the time nodes on the schedule, the direction at which the agent travels
            size: self.num_of_time_samples + 1
            source: +1 or -1 according to the flip schedule
        """
        # generate the flip schedule: at those time nodes the agent flips direction
        flip_schedule = []
        global_time = 0
        while global_time < self.endtime:
            a_time_interval = np.random.exponential(1/self.F)
            flip_schedule.append(a_time_interval)
            global_time += a_time_interval
        flip_schedule = np.cumsum(flip_schedule) # does not include 0, overshoots the endtime

        # merge with the sample nodes for the master schedule
        sample_nodes = self.timenodes # include 0, ends right at the endtime

        master_schedule = np.sort(np.concatenate((flip_schedule, sample_nodes), axis = 0))
        if master_schedule[-1] > self.endtime:
            master_schedule = master_schedule[:-1] # delete the last overshoot elm

        # create the direction table
        directions = np.zeros_like(master_schedule)
        flip_schedule_pointer = 0
        local_direction = 2*(np.random.rand()>0.5)-1 # direction, left or right (-1 or 1), random initial
        for index, time_node in enumerate(master_schedule):
            if time_node == flip_schedule[flip_schedule_pointer]:
                local_direction *= -1
                flip_schedule_pointer += 1

            directions[index] = local_direction

        return master_schedule, directions

    def _setup_time_evolution_plot_vars(self, Ntimenodes):
        """
        Set up variables for time evolutioin plot
        """
        # how many times we will sample from 0 to self.endtime
        self.num_of_time_samples = Ntimenodes
        # at those nodes we record the loc of each agent, including 0
        self.timenodes = np.linspace(0, self.endtime, self.num_of_time_samples + 1, endpoint = True)
        # storaging loc info at each time node: row, agent index; col: time node index
        self.locmatrix = np.zeros((self.iterations, self.num_of_time_samples + 1))
        # direction info at each time node
        self.dirmatrix = np.zeros_like(self.locmatrix)


    def _remove_value_from_list(self, value, alist):
        """
        Removes a given value from a list
        -----
        Returns:
            the resulted list
        """
        return [local_val for local_val in alist if local_val != value]

    #@numba.vectorize #('float64(int32, float64, float64)', target='parallel')
    def _trueloc(self, direction, location, traveltime):
        """
        Compute the true location given current direction, location and travel time

        direction: 1 or -1
        location: in x variable
        traveltime: in seconds
        -----
        Returns:
            next location in x given all the data
        """
        yloc = self.y_from_x(location) # change into y coord
        yloc = (yloc + traveltime*direction) % self.yupper # compute final location in y then mod back into the domain
        xloc = self.x_from_y(yloc) # change back into x

        return xloc

    def s_y(self, y):
        """
        s(y) := s(x(y))
        """
        return self.s(self.x_from_y(y))

    def _build_y_from_x_interp(self, datapoint = 10001):
        """
        Get y from x using integration, then interpolate for faster speed
        About half as fast, good enough
        """
        xdata = np.linspace(0, self.domainsize + self.epsilon, datapoint, endpoint=True)
        y_theoretical = self._y_from_x_theoretical
        ydata = list(map(y_theoretical, xdata))
        y_polyinterp = interp1d(xdata, ydata, kind = 'linear')
        x_polyinterp = interp1d(ydata, xdata, kind = 'linear')
        return x_polyinterp, y_polyinterp

    def _y_from_x_theoretical(self, x):
        func = lambda x: 1/self.s(x)
        y = quad(func, 0, x)[0] + self.ylower

        return y

    def _setup_s(self, s):
        """
        Default speed is cos(2*pi*x) + 1.5
        """
        if s is None:
            return lambda x: np.cos(x*(2*np.pi)) + 1.5
        else:
            return s

    def _setup_ylu(self):
        """
        lowerbound and upperbound in y
        """
        self.ylower = 0
        func = lambda x: 1/self.s(x)
        self.yupper = quad(func, 0, self.domainsize)[0] + self.ylower
        self.yupper_hidden = quad(func, 0, self.domainsize + self.epsilon)[0] + self.ylower


#%% tests
# MC = MC_Stationary_Toxin_Cos(Ntimenodes = 200, endtime=10, iterations = 10000)
# MC.run_all_agents()
#  #%% simple grapher
# bins = np.linspace(0, MC.domainsize, 100, endpoint = True)
# MC.hist_plot(0, bins)

# #%% preliminary graphers for checking the MC model
# plt.style.use('ggplot')
# # graph s(x) and y(x), interpolated y(x)
# xdata = np.linspace(0, MC.domainsize, 1000)

# fig1 = plt.figure(dpi = 300)
# ax1 = fig1.add_subplot(111)

# ax1.plot(xdata, MC.s(xdata), label = r'$s(x)$')
# ax1.plot(xdata, list(map(MC._y_from_x_theoretical, xdata)), label = r'$y(x)$')
# ax1.plot(xdata, MC.y_from_x(xdata), linestyle = '--', label = r'itp $y(x)$')

# ax1.legend(loc = 'best')
# fig1.show()

# # graph s(y), x(y) interpolated and y(x)
# ydata = np.linspace(MC.ylower, MC.yupper, 1000)

# fig2 = plt.figure(dpi = 300)
# ax2 = fig2.add_subplot(111)

# ax2.plot(ydata, MC.s_y(ydata), label = r'$s(y)$')
# ax2.plot(ydata, MC.x_from_y(ydata), label = r'$x(y)$')
# ax2.legend(loc = 'best')
# fig2.show()

# # graph x(y) interpolated and y(x)
# fig3 = plt.figure(dpi = 300)
# ax3 = fig3.add_subplot(111)

# ax3.plot(ydata, MC.x_from_y(ydata), label = r'$x(y)$')
# ax3.plot(xdata, MC.y_from_x(xdata), label = r'$y(x)$')
# ax3.legend(loc = 'best')

# ax3.set_xlim(0,1.1)
# ax3.set_ylim(0,1.1)
# ax3.set_aspect('equal')
# fig3.show()
# #%% Save MC result
# np.save(r'D:\Research with Dr Rossi\Numerical Plot Images\Plot MC cos speed\MC_data', MC.locmatrix)

# #%% use numpy to calculate density/histogram from MC data
# import os
# os.chdir(r'D:\Research with Dr Rossi\Numerical Plot Images\Plot MC cos speed')
# MC_locmatrix =  np.load(r'D:\Research with Dr Rossi\Numerical Plot Images\Plot MC cos speed\MC_data.npy')
# MC.locmatrix = MC_locmatrix
# #%%
# fig, ax  = plt.subplots(dpi = 300)
# #ax = fig.add_subplot(111)
# bins = np.linspace(0, 1, 101, endpoint = True)
# hist, bin_edges = np.histogram(MC_locmatrix[:, 3], bins = bins, density=True)
# ax.bar(bins[:-1]+1/2*(bins[1]-bins[0]), hist,  alpha = 0.5,  width =(bins[1]-bins[0]))
# ax.set_ylim([0,4])
# ax.title.set_text(r'time = {:.2f}s'.format(3/20))
# fig.show()

# # plt.ioff()  # uncomment to show figures
# # for node in range(MC_locmatrix.shape[1]): #(MC.locmatrix.shape[1]):
# #     bins = np.linspace(0, 1, 101, endpoint = True)
# #     hist, bin_edges = np.histogram(MC_locmatrix[:, node], bins = bins, density=True)

# #     fig = plt.figure(dpi = 300)
# #     ax = fig.add_subplot(111)
# #     ax.bar(bins[:-1]+1/2*(bins[1]-bins[0]), hist,  alpha = 0.5,  width =(bins[1]-bins[0]))
# #     ax.set_ylim([0,4])
# #     ax.title.set_text(r'time = {:.2f}s'.format(node/20))
# #     fig.savefig('node_' + str(node))
# #     plt.close()

# #%% create density from MC at 0s, 0.1s, 0.15s, 0.5s, 10s
# bins = np.linspace(0, MC.domainsize, 101, endpoint = True)
# hist0, bin_edges = np.histogram(MC.locmatrix[:,0], bins = bins, density=True)
# hist1, _ = np.histogram(MC.locmatrix[:,2], bins = bins, density=True)
# hist1p5, _ = np.histogram(MC.locmatrix[:,3], bins = bins, density=True)
# hist5, _ = np.histogram(MC.locmatrix[:,10], bins = bins, density=True)
# hist10, _ = np.histogram(MC.locmatrix[:,20], bins = bins, density=True)
# hist100, _ = np.histogram(MC.locmatrix[:,200], bins = bins, density=True)

# x_data = bins[:-1]+1/2*(bins[1]-bins[0])
# #%% save MC result
# import pandas as pd
# final_time_data = {'xdata': x_data,
#                    'density0s': hist0,
#                    'density0p1s': hist1,
#                    'density0p15s': hist1p5,
#                    'density0p5s': hist5,
#                    'density1s': hist10,
#                    'density10s': hist100}
# df = pd.DataFrame(final_time_data)
# df.to_csv (r'D:\Research with Dr Rossi\Python Scratch\Plotting for 1st paper\MC_stationary_toxin_data.csv', index = False, header=True)
# #%% load MC result
# import matplotlib.pyplot as plt
# import numpy as np

# import pandas as pd
# data = pd.read_csv(r'D:\Research with Dr Rossi\Python Scratch\Plotting for 1st paper\MC_stationary_toxin_data.csv')
# x_data = data['xdata'].to_numpy()
# density0s = data['density0s'].to_numpy()
# density0p1s = data['density0p1s'].to_numpy()
# density0p15s = data['density0p15s'].to_numpy()
# density0p5s = data['density0p5s'].to_numpy()
# density1s = data['density1s'].to_numpy()
# density10s = data['density10s'].to_numpy()

# #%% load analytical result
# import pandas as pd
# data = pd.read_csv(r'D:\Research with Dr Rossi\Mathematica Scratch\Fourier Series Sol Stationary Toxin Cosine.csv')
# x_data_aly = data[r'xdata'].to_numpy()
# density0s_aly = data[r'Sig0'].to_numpy()
# density0p1s_aly = data[r'Sig0.1'].to_numpy()
# density0p15s_aly = data[r'Sig0.15'].to_numpy()
# density0p5s_aly = data[r'Sig0.5'].to_numpy()
# density1s_aly = data[r'Sig1'].to_numpy()
# density10s_aly = data[r'Sig10'].to_numpy()

# #%% Plot analytical data against MC data

# #subplots: beginning, middle 1, middle 2, end
# plt.style.use('seaborn-paper')
# linewidth = 4
# linestyle = '--'
# bar_width = x_data[1]-x_data[0]
# #plt.rcParams.update({'font.size': 7})
# fig, ax = plt.subplots(2,2, figsize=(20, 12), dpi = 100)
# all_ax = [ax[0,0], ax[0,1], ax[1,0], ax[1,1]]
# ax[0, 0].bar(x_data, density0s, color='steelblue', alpha=0.6, width = bar_width,
#              label = 'Simulated $\Sigma$')
# ax[0,0].plot(x_data_aly, density0s_aly, color = 'black', linestyle = linestyle, linewidth=linewidth,
#              label = 'Analytical $\Sigma$')
# ax[0,0].set_ylim(0,3)
# #ax[0,0].set_title('t=0', fontsize=30)
# fig.tight_layout(pad=7)
# xtick = [0,0.5,1]
# ytick = [0,1,2,3,4]

# ax[1,0].bar(x_data, density0p5s, color='steelblue', alpha=0.6, width = bar_width)
# ax[1,0].plot(x_data_aly, density0p5s_aly, color = 'black', linestyle = linestyle, linewidth=linewidth)
# ax[1,0].set_ylim(0,3)
# #ax[1,0].set_title('t=0.5', fontsize=30)

# ax[0,1].bar(x_data, density0p15s, color='steelblue', alpha=0.6, width = bar_width)
# ax[0,1].plot(x_data_aly, density0p15s_aly, color = 'black', linestyle = linestyle, linewidth=linewidth)
# ax[0,1].set_ylim(0,3)
# #ax[0,1].set_title('t=0.15', fontsize=30)

# ax[1, 1].bar(x_data, density10s, color='steelblue', alpha=0.6, width = bar_width)
# ax[1,1].plot(x_data_aly, density10s_aly, color = 'black', linestyle = linestyle, linewidth=linewidth)
# ax[1,1].set_ylim(0,3)
# #ax[1,1].set_title('t=10', fontsize=30)

# # Time
# fig.text(0.4, 0.85, 't=0', va='center', fontsize=30)
# fig.text(0.85, 0.85, 't=0.15', va='center', fontsize=30)
# fig.text(0.4, 0.4, 't=0.5', va='center', fontsize=30)
# fig.text(0.85, 0.4, 't=10', va='center', fontsize=30)

# # 'x' label
# fig.text(0.25, 0.5, '$x$', va='center', fontsize=30)
# fig.text(0.73, 0.5, '$x$', va='center', fontsize=30)
# fig.text(0.25, 0.03, '$x$', va='center', fontsize=30)
# fig.text(0.73, 0.03, '$x$', va='center', fontsize=30)

# # 'Sigma' label
# fig.text(0.02, 0.75, r'$\Sigma$', va='center', fontsize=30)
# fig.text(0.50, 0.75, r'$\Sigma$', va='center', fontsize=30)
# fig.text(0.02, 0.30, r'$\Sigma$', va='center', fontsize=30)
# fig.text(0.50, 0.30, r'$\Sigma$', va='center', fontsize=30)

# # ABCD label
# fig.text(0.08, 0.88, r'$\it{(a)}$', va='center', fontsize=40)
# fig.text(0.56, 0.88, r'$\it{(b)}$', va='center', fontsize=40)
# fig.text(0.08, 0.42, r'$\it{(c)}$', va='center', fontsize=40)
# fig.text(0.56, 0.42, r'$\it{(d)}$', va='center', fontsize=40)

# # fig.text(0.02, 0.5, 'Density', va='center', rotation='vertical', fontsize=30)
# # fig.text(0.45, 0.02, 'Location', va='center', fontsize=30)
# fig.legend(bbox_to_anchor=(0.08,0.82), loc="upper left", borderaxespad=0, fontsize=25)

# for each_ax in all_ax:
#     each_ax.tick_params(axis='both', which='major', labelsize=28)
#     each_ax.set_xticks(xtick)
#     each_ax.set_yticks(ytick)
# fig.show()

# #%% Plot analytical sol against MC steady state
# #subplots: beginning, middle 1, middle 2, end
# plt.style.use('seaborn-paper')
# linewidth = 1
# linestyle = '--'
# bar_width = x_data[1]-x_data[0]
# #plt.rcParams.update({'font.size': 7})
# fig, ax = plt.subplots(1,1, figsize=(8, 5), dpi = 200)
# ax.bar(x_data, density10s, color='steelblue', alpha=0.6, width = bar_width,
#              label = 'Simulated $\Sigma$')
# ax.plot(x_data_aly, density10s_aly, color = 'black', linestyle = linestyle, linewidth=linewidth,
#              label = 'Analytical $\Sigma$')
# ax.set_ylim(0,3)
# ax.set_xlabel("Location")
# ax.set_ylabel("Density")
# ax.legend()
# ax.set_title('Steady State Density, s0=0', fontsize=10)
# fig.tight_layout(pad=7)
# xtick = [0,0.5,1]
# ytick = [0,1,2,3,4]

# fig.show()

# #%% Plot y(x) in terms of s(x), s(x) being linear and cos(2*pi*x) + 1.5
# import numpy as np
# from scipy.integrate import quad
# import matplotlib.pyplot as plt

# # 1. Define s(x)
# def s_cos(x: float):
#     speed = np.cos(x * 2 * np.pi) + 1.5
#     return speed

# def y(x):
#     s_inv = lambda x: 1 / s_cos(x)
#     y_value = quad(s_inv, 0, x)[0]
#     return y_value
# y_vec = np.vectorize(y)
# #%%
# # 2. Generate data
# x_data = np.linspace(1e-9, 1, 3001)
# y_data = y_vec(x_data)

# # 3. Plot
# fig, ax = plt.subplots(2, 1, figsize=(10, 5), dpi = 200)
# ax[0].plot(x_data, y_data, label='y(x)')
# ax[0].set_xticks(ticks=[0, 0.25, 0.5, 0.75, 1])   # fontsize of the tick labels
# ax[0].set_xticklabels(labels=[0, 0.25, 0.5, 0.75, 1], Fontsize=15)
# ax[0].set_yticks(ticks = [0, 0.25, 0.5, 0.75, 1])   # fontsize of the tick labels
# ax[0].set_yticklabels(labels=[0, 0.25, 0.5, 0.75, 1], Fontsize=15)
# ax[0].legend(bbox_to_anchor=(0.16, 0.98), fontsize=15)
# ax[0].set_xlabel('x', fontsize=15)
# ax[0].set_xlim(-0.02, 1.02)
# ax[0].grid()

# ax[1].plot(x_data, s_cos(x_data), label='$s(x) = \cos(2 \pi x) + 1.5 $')
# ax[1].set_xticks(ticks = [0, 0.25, 0.5, 0.75, 1])   # fontsize of the tick labels
# ax[1].set_xticklabels(labels=[0, 0.25, 0.5, 0.75, 1], Fontsize=15)
# ax[1].set_yticks(ticks = [0.5, 1, 1.5, 2, 2.5])   # fontsize of the tick labels
# ax[1].set_yticklabels(labels=[0.5, 1, 1.5, 2, 2.5], Fontsize=15)
# ax[1].legend(bbox_to_anchor=(0.3478, 0.30), fontsize=15)
# ax[1].set_xlabel('x', fontsize=15)
# ax[1].set_xlim(-0.02, 1.02)
# ax[1].grid()

# plt.tight_layout()
# plt.show()
