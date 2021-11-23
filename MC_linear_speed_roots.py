# -*- coding: utf-8 -*-
"""
Monte Carlo, s-1 has roots and is piecewise linear

@author: Hansen
"""

import numpy as np
import matplotlib.pyplot as plt
import time


class MC_Linear_Speed_Roots:
    def __init__(self,
                 s = None,
                 F = 1,
                 iterations = 3000,  # num of agents
                 endtime = 20,  # when the simulation ends
                 domainsize = 1,  # x from 0 to domainsize
                 Ntimenodes = 200  # stores location and direction info at those nodes, not including t = 0
                 ):
        # MC parameters
        self.iterations = iterations
        self.endtime = endtime
        self.epsilon = 1E-3  # Param for expanding x on the right end for calculating y
        # Model parameters
        self.domainsize = domainsize
        self.s = self._setup_s(s)
        self.F = F
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
            # For P+, use domain mapping in subintervals, follow s-1
            if directions[cur_step-1] == 1:
                loc[cur_step] = self._trueloc_p(direction = directions[cur_step-1],
                                                location = loc[cur_step-1],
                                                traveltime = schedule[cur_step] - schedule[cur_step-1])
            else:  # For P-, use domain mapping, follow -s-1
                loc[cur_step] = self._trueloc_m(direction = directions[cur_step-1],
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
        # xdata = np.linspace(0,self.domainsize, 1001)
        # inv_s_data = list(map(lambda x: 1/self.s(x)/self.yupper, xdata))
        # ax.plot(xdata, inv_s_data, label = r'$1/s(x)$')

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

    def _trueloc_m(self, direction, location, traveltime):
        """
        Compute the true location given current direction, location and travel time
        This is only for P-. Also everything is hardcoded for accuracy.

        direction: -1 for this
        location: in x variable
        traveltime: in seconds
        -----
        Returns:
            next location in x given all the data
        """
        xloc = location
        y_upper = np.log(361/225)
        # change into y coord
        yloc = ((-np.log(19 - 8*xloc) + np.log(19)) * (0 <= xloc <= 1/2)
                + (np.log(19/225) + np.log(11 + 8*xloc)) * (1/2 < xloc <= 1))
        # compute final location in y then mod back into the domain
        yloc = (yloc + traveltime*direction) % y_upper
        # change back into x
        xloc_new = (((1 - np.exp(-yloc))*19/8) * (0 <= yloc <= np.log(19/15))
                    + (225*np.exp(yloc)/8/19 - 11/8) * (np.log(19/15) < yloc <= y_upper))

        return xloc_new

    @staticmethod
    def _xtoy1(xloc):
        """
        Mapping from x to y in [0, 3/8]
        """
        yloc = -np.log((3 - 8*xloc) / 3)
        return yloc

    @staticmethod
    def _xtoy2(xloc):
        """
        Mapping from x to y in [3/8, 1/2]
        """
        yloc = -np.log(8*xloc - 3)
        return yloc

    @staticmethod
    def _xtoy3(xloc):
        """
        Mapping from x to y in [1/2, 5/8]
        """
        yloc = np.log(5 - 8*xloc)
        return yloc

    @staticmethod
    def _xtoy4(xloc):
        """
        Mapping from x to y in [5/8, 1]
        """
        yloc = np.log((8 * xloc - 5) / 3)
        return yloc

    @staticmethod
    def _ytox1(yloc):
        """
        Mapping from y to x for x in [0, 3/8]
        """
        xloc = (3 - 3 * np.exp(-yloc)) / 8
        return xloc

    @staticmethod
    def _ytox2(yloc):
        """
        Mapping from y to x for x in [3/8, 1/2]
        """
        xloc = (np.exp(-yloc) + 3) / 8
        return xloc

    @staticmethod
    def _ytox3(yloc):
        """
        Mapping from y to x for x in [1/2, 5/8]
        """
        xloc = (-np.exp(yloc) + 5) / 8
        return xloc

    @staticmethod
    def _ytox4(yloc):
        """
        Mapping from y to x for x in [5/8, 1]
        """
        xloc = (3*np.exp(yloc) + 5) / 8
        return xloc

    def _trueloc_p(self, direction, location, traveltime):
        """
        Compute the true location given current direction, location and travel time
        This is only for P+ and is hardcoded

        direction: 1 for this
        location: in x variable
        traveltime: in seconds
        -----
        Returns:
            next location in x given all the data
        """

        # 1. Determine if it is inside or outside of (x1, x3)
        xloc = location
        # Translate x to y, for x outside (x1, x3)
        if (xloc < 3/8 or xloc > 5/8):
            yloc = self._xtoy1(xloc) if xloc < 3/8 else self._xtoy4(xloc)
            # Advance in time
            yloc += direction*traveltime
            # Tranlsate y to x
            xloc_new = self._ytox1(yloc) if yloc >= 0 else self._ytox4(yloc)

        # Translate x to y, for x inside (x1, x3)
        elif (xloc > 3/8 and xloc < 5/8):
            yloc = self._xtoy2(xloc) if xloc <= 1/2 else self._xtoy3(xloc)
            # Advance in time
            yloc += direction*traveltime
            # Tranlsate y to x
            xloc_new = self._ytox2(yloc) if yloc >= 0 else self._ytox3(yloc)

        # For x == x1, x3, do not move
        else:
            xloc_new = xloc

        return xloc_new

    def _setup_s(self, s):
        """
        Default speed is (-(x-x1)+1)*(0<=x<=0.5) + ((x-x3)+1)*(0.5<x<=1)
        """
        if s is None:
            return lambda x: (-(x-self.x1)+1)*(0<=x<=0.5) + ((x-self.x3)+1)*(0.5<x<=1)
        else:
            return s


#%% Sanity Checks
# MC = MC_Linear_Speed_Roots()
# # print(MC._xtoy1(1/4), MC._xtoy2(7/16), MC._xtoy3(9/16), MC._xtoy4(3/4))
# # print(MC._ytox1(1), MC._ytox2(1), MC._ytox3(-1), MC._ytox4(-1))
# print(MC._trueloc_p(1, 5/8-0.01, 2))
