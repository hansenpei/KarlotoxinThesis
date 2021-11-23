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


#%% Sanity Check
# MC = MC_Stationary_Toxin_Cos(Ntimenodes = 200, endtime=10, iterations = 10000)
# MC.run_all_agents()
