# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 18:10:31 2021

@author: Hansen
"""
import numpy as np
import matplotlib.pyplot as plt

class FiniteDiff:
    """
    Default uses Lax Friedrich
    """
    def __init__(self, s,
                 gamma = None,
                 x1 = 0.19884821244195336,
                 x3 = 0.6383641464795798,
                 CFL = 0.2,
                 N = 201,
                 F = 1,
                 s0 = 1,
                 T = 5):
        # scheme parameters
        self.CFL = CFL
        self.N = N
        self.T = T
        self.dx = 1 / N
        self.dt = CFL*self.dx  # CFL = dt / dx
        self.Nt = int(self.T // self.dt)
        self.time_step = 0 # initialize time step
        # model parameters
        self.F = F
        self.s = s # function
        self.s0 = s0
        self.x1 = x1
        self.x2 = gamma
        self.x3 = x3
        self._adjust_CFL()
        self._set_up_P_x_t()
        self._find_index_x1_x3()

    def run(self, end_step = None, if_plot = True, if_show_progress = True):
        if end_step is None:
            end_step = self.Nt
        """
        Run the LF method

        Parameters
        ----------
        end_step : int, optional
            How many steps to go. The default is self.Nt.
        if_plot : bool, optional
            Whether to plot at the end. The default is True.
        if_show_progress : bool, optional
            Whether show progress. The default is True.

        Returns
        -------
        None.

        """
        # stuff needed for displaying progresses
        if if_show_progress:
            display_nodes = [self.Nt*i//100 for i in range(1,100,5)] + [self.Nt]

        # actual stepping
        self.time_step = 0 # initialize time step
        milestone = 0
        for n in range(end_step + 1): # wanna include the end_step
            self.step()
            if if_show_progress and (n == display_nodes[milestone]):
                print("{:.1f} percent finished".format(n/self.Nt *100))
                milestone += 1

        # plotting
        self._calculate_Sig_Delt_mu_xi()
        if if_plot == True:
            #self._calculate_Sig_Delt_mu_xi()

            fig = plt.figure(dpi = 300)
            ax = fig.add_subplot(1,1,1)
            x_vec_plt = self.x_vec
            ax.plot(x_vec_plt, self.Pp[:, self.time_step], label='P+')
            ax.plot(x_vec_plt, self.Pm[:, self.time_step], label='P-')
            ax.plot(x_vec_plt, self.Sigma[:, self.time_step], label='$\Sigma$')
            ax.plot(x_vec_plt, self.Delta[:, self.time_step], label='$\Delta$')
            ax.plot(x_vec_plt, self.xi[:, self.time_step], label='$\\xi$')
            ax.plot(x_vec_plt, self.eta[:, self.time_step], label='$\eta$')
            ax.legend(loc = 'best')
            fig.tight_layout()
            fig.show()

        print("Nt is {}, current time step is {}".format(self.Nt, self.time_step))

    def step(self):
        """
        Using the Lax-Friedrichs scheme:
            P+(x, t+dt) = 1/2 [P+(x+dx, t) + P+(x-dx, t)] - CFL/2*[M+(x+dx, t) - M+(x-dx, t)]
                        + F[P-(x, t) - P+(x, t)] dt
            P-(x, t+dt) = 1/2 [P-(x+dx, t) + P-(x-dx, t)] - CFL/2*[M-(x+dx, t) - M-(x-dx, t)]
                        + F[P+(x, t) - P-(x, t)] dt
        """
        if self.time_step < (self.Nt-1):
            n = self.time_step
            s = self.s
            x = self.x_vec
            idx = self.idx
            s0 = self.s0
            # spatial loop
            for j in range(0, self.N):
                jl = idx(j - 1)
                jr = idx(j + 1)
                # Pp
                self.Pp[j,n+1] = (
                    1/2 * (self.Pp[jr, n] + self.Pp[jl, n]) - self.dt/self.dx/2*(
                        (s(x[jr])-s0)*self.Pp[jr,n] - (s(x[jl])-s0)*self.Pp[jl,n])
                    + self.F * (self.Pm[j,n] - self.Pp[j,n]) * self.dt)

                # Pm
                self.Pm[j,n+1] = (
                    1/2 * (self.Pm[jr, n] + self.Pm[jl, n]) - self.dt/self.dx/2*(
                        (-s(x[jr])-s0)*self.Pm[jr,n] - (-s(x[jl])-s0)*self.Pm[jl,n])
                    + self.F * (self.Pp[j,n] - self.Pm[j,n]) * self.dt)

            self.time_step += 1
        return None

    def idx(self, n):
        """
        Wrap the index around for periodic BC
        """

        return n % self.N


    def _set_up_P_x_t(self):
        """
        Initiate x_vec, t_vec, s_col, Pp, Pm
        """
        # y is evenly spaced, then mapped to x
        self.x_vec = np.linspace(0, 1, self.N, endpoint=False)
        self.t_vec = np.array([n*self.dt for n in range(self.Nt)])
        # initialize but no need to put in values until later
        self.Pp = np.zeros((self.N, self.Nt))
        self.Pm = np.zeros_like(self.Pp)
        self.Sigma = np.zeros_like(self.Pp)
        self.Delta = np.zeros_like(self.Pp)
        self.xi = np.zeros_like(self.Pp)
        self.eta = np.zeros_like(self.Pp)
        self.Pp[:,0] = 0.5
        self.Pm[:,0] = 0.5
        self.s_col = np.array([self.s(self.x_vec)[:]]).T # column vector of s data

    def _calculate_Sig_Delt_mu_xi(self, time_step = None):
        """
        Calculate Sigma, Delta, mu, xi,
         from time step 0 to time_step included;
        time_step is default to the current timestep, i.e., self.time_step
        """
        if time_step is None:
            time_step = self.time_step + 1
        self.Sigma[:,:time_step] = self.Pp[:,:time_step] + self.Pm[:,:time_step]
        self.Delta[:,:time_step] = self.Pp[:,:time_step] - self.Pm[:,:time_step]
        self.xi[:,:time_step] = self.s_col*self.Sigma[:,:time_step] - self.Delta[:,:time_step]
        self.eta[:,:time_step] = self.s_col*self.Delta[:,:time_step] - self.Sigma[:,:time_step]

    def _adjust_CFL(self):
        """
        Adjust dt and dx to meet the CFL condition
        alpha * dt / dx < 1, alpha = max flux speed.
        """

        # find the max_x s'(x) = alpha
        alpha = self._max_slope_of_flux()

        # construct an alternative CFL
        if self.CFL*alpha > 1:
            print("CFL is changed from {} to {}".format(self.CFL, 1/(alpha + 0.5)))
            self.CFL = 1/(alpha + 0.5)
            self.dt = self.CFL*self.dx
            self.Nt = int(self.T // self.dt)

    def _max_slope_of_flux(self, grid = 10000):
        """
        Find the maximum of the flux derivative term:
            |s(x) - s0|, |s(x) + s0|
            throughout x to adjust CFL
        """

        # find max flux speed numerically
        x_vec = np.linspace(0, 1, grid, endpoint=False)
        flux_vec1 = np.abs(self.s(x_vec) - self.s0)
        flux_vec2 = np.abs(self.s(x_vec) + self.s0)
        max_flux_speed = np.max((flux_vec1, flux_vec2))
        return max_flux_speed

    def _find_index_x1_x3(self):
        """
        Find the largest indices j1, j3 on the stencil that are closest to x1 and x3 respectively.
        """
        x1 = self.x1
        x3 = self.x3
        x_vec = self.x_vec
        temp_j1_1 = len(x_vec[x_vec<=x1]) - 1
        temp_j1_2 = temp_j1_1 + 1
        temp_j3_1 = len(x_vec[x_vec<=x3]) - 1
        temp_j3_2 = temp_j3_1 + 1

        self.j1 = temp_j1_1 if abs(x_vec[temp_j1_1] - x1) < abs(x_vec[temp_j1_2] - x1) else temp_j1_2
        self.j3 = temp_j3_1 if abs(x_vec[temp_j3_1] - x3) < abs(x_vec[temp_j3_2] - x3) else temp_j3_2
        self.j1l = temp_j1_1  # stencil closest to x3, on the left
        self.j1r = temp_j1_2  # stencil closest to x3, on the right


class RichtmyerLaxWendroff(FiniteDiff):

    def _mid_x(self, i1, i2):
        """
        Find the mid value of the x given two index
        Force i1 < i2
        """

        idx = (i1 + i2) / 2
        mid_x = idx * self.dx % 1

        return mid_x

    def step(self):
        """
        Using the Richtmyer two-step Lax-Wendroff scheme:
            Pp_half_l = 1/2 * [Pp(x, t) + Pp(x-dt, t)] - dt/dx/2 [Mp(x, t) - Mp(x-dx, t)] + source/2
            Pp_half_r = 1/2 * [Pp(x+dt, t) + Pp(x, t)] - dt/dx/2 [Mp(x+dx, t) - Mp(x, t)] + source/2

            Pm_half_l = 1/2 * [Pm(x, t) + Pm(x-dt, t)] - dt/dx/2 [Mm(x, t) - Mm(x-dx, t)] + source/2
            Pm_half_r = 1/2 * [Pm(x+dt, t) + Pm(x, t)] - dt/dx/2 [Mm(x+dx, t) - Mm(x, t)] + source/2

            Pp(x, t+dt) = Pp(x, t) - dt/dx*[sp(x+dx)*Pp_half_r - sp(x-dx)*Pp_half_l]
                        + F[Pm(x, t) - Pp(x, t)] dt
            Pm(x, t+dt) = Pm(x, t) - dt/dx*[sm(x+dx)*Pm_half_r - sm(x-dx)*Pm_half_l]
                        + F[Pp(x, t) - Pm(x, t)] dt
        """
        if self.time_step < (self.Nt-1):
            n = self.time_step
            s = self.s
            x = self.x_vec
            idx = self.idx
            s0 = self.s0
            # spatial loop
            for j in range(0, self.N):
                # Half step values
                jl = idx(j-1)
                jr = idx(j+1)
                Pp_half_l = ((self.Pp[j, n] + self.Pp[jl, n])
                    - self.dt/self.dx * ((s(x[j])-s0)*self.Pp[j,n] - (s(x[jl])-s0)*self.Pp[jl,n])
                    + self.F * (self.Pm[j,n] - self.Pp[j,n] + self.Pm[jl,n] - self.Pp[jl,n]) * self.dt / 2) / 2
                Pp_half_r = ((self.Pp[jr, n] + self.Pp[j, n])
                    - self.dt/self.dx * ((s(x[jr])-s0)*self.Pp[jr,n] - (s(x[j])-s0)*self.Pp[j,n])
                    + self.F * (self.Pm[jr,n] - self.Pp[jr,n] + self.Pm[j,n] - self.Pp[j,n]) * self.dt / 2) / 2
                Pm_half_l = ((self.Pm[j, n] + self.Pm[jl, n])
                    - self.dt/self.dx * ((-s(x[j])-s0)*self.Pm[j,n] - (-s(x[jl])-s0)*self.Pm[jl,n])
                    + self.F * (self.Pp[j,n] - self.Pm[j,n] + self.Pp[jl,n] - self.Pm[jl,n]) * self.dt / 2) / 2
                Pm_half_r = ((self.Pm[jr, n] + self.Pm[j, n])
                    - self.dt/self.dx * ((-s(x[jr])-s0)*self.Pm[jr,n] - (-s(x[j])-s0)*self.Pm[j,n])
                    + self.F * (self.Pp[jr,n] - self.Pm[jr,n] + self.Pp[j,n] - self.Pm[j,n]) * self.dt / 2) / 2
                s_half_l = s(self._mid_x(j-1, j))
                s_half_r = s(self._mid_x(j, j+1))
                # Pp
                self.Pp[j,n+1] = (self.Pp[j,n]
                    - self.dt/self.dx*((s_half_r - s0) * Pp_half_r - (s_half_l - s0) * Pp_half_l)
                    + self.F * (self.Pm[j,n] - self.Pp[j,n]) * self.dt)
                # Pm
                self.Pm[j,n+1] = (self.Pm[j,n]
                    - self.dt/self.dx*((-s_half_r - s0) * Pm_half_r - (-s_half_l - s0) * Pm_half_l)
                    + self.F * (self.Pp[j,n] - self.Pm[j,n]) * self.dt)

            self.time_step += 1
        return None

class UpwindLF(FiniteDiff):
    """
    Use LF flux near x3 (Method proposed in original thesis)
    """

    def step(self):
        """
        Advance one time step.
        """
        n = self.time_step
        s = self.s
        x = self.x_vec
        idx = self.idx
        s0 = self.s0
        dt_dx = self.dt / self.dx
        Pp = self.Pp
        Pm = self.Pm
        if self.time_step < (self.Nt-1):
            n = self.time_step
            # spatial loop
            for j in range(0, self.N):
                jl = idx(j-1)
                jr = idx(j+1)
                # Pp
                if j < self.j1 or j > self.j3:  # Positive flux, upwind is left
                    Pp[j,n+1] = Pp[j,n] - dt_dx*( (s(x[j])-s0)*Pp[j,n] - (s(x[jl])-s0)*Pp[jl,n] ) \
                        + self.dt*(Pm[j,n] - Pp[j,n])*self.F
                elif j > self.j1 and j < self.j3:  # Negative flux, upwind is right
                    Pp[j,n+1] = Pp[j,n] - dt_dx*( (s(x[jr])-s0)*Pp[jr,n] - (s(x[j])-s0)*Pp[j,n] ) \
                        + self.dt*(Pm[j,n] - Pp[j,n])*self.F
                elif j == self.j1: # At x1, take flux from both directions
                    Pp[j,n+1] = Pp[j,n] - dt_dx*( (s(x[jr])-s0)*Pp[jr,n] - (s(x[jl])-s0)*Pp[jl,n] ) \
                        + self.dt*(Pm[j,n] - Pp[j,n])*self.F
                elif j == self.j3:  # At x3, Lax-Friedrichs
                    Pp[j,n+1] = Pp[j,n] - dt_dx/2*( (s(x[jr])-s0)*Pp[jr,n] - (s(x[jl])-s0)*Pp[jl,n] ) \
                        + 1/2*(Pp[j+1,n] + Pp[j-1,n] - 2*Pp[j,n]) \
                        + self.dt*(Pm[j,n] - Pp[j,n])*self.F
                # elif j == self.j3+1:
                #     self.Pp[j,n+1] = self.Pp[j,n] - self.CFL*( (s(x[j])-s0)*self.Pp[j,n] - (s(x[idx(j-1)])-s0)*self.Pp[idx(j-1),n] ) \
                #         + self.dt*(self.Pm[j,n] - self.Pp[j,n]) \
                #         + eps*(-self.Pp[j+1,n]- self.Pp[j-1,n] + 2*self.Pp[j,n])*self.CFL
                else:
                    raise ValueError('flag')
                # Pm
                Pm[j,n+1] = Pm[j,n] - dt_dx*( (-s(x[jr])-s0)*Pm[jr,n] - (-s(x[j])-s0)*Pm[j,n] ) \
                        + self.dt*(Pp[j,n] - Pm[j,n])*self.F

            self.time_step += 1
        return None

class UpwindLFHalfSink(FiniteDiff):
    """
    Use LF flux near x3 (Source), Half coefficient at x1 (Sink)
    """

    def step(self):
        """
        Advance one time step.
        """
        n = self.time_step
        s = self.s
        x = self.x_vec
        idx = self.idx
        s0 = self.s0
        dt_dx = self.dt / self.dx
        Pp = self.Pp
        Pm = self.Pm
        if self.time_step < (self.Nt-1):
            n = self.time_step
            # spatial loop
            for j in range(0, self.N):
                jl = idx(j-1)
                jr = idx(j+1)
                # Pp
                if j < self.j1 or j > self.j3:  # Positive flux, upwind is left
                    Pp[j,n+1] = Pp[j,n] - dt_dx*( (s(x[j])-s0)*Pp[j,n] - (s(x[jl])-s0)*Pp[jl,n] ) \
                        + self.dt*(Pm[j,n] - Pp[j,n])*self.F
                elif j > self.j1 and j < self.j3:  # Negative flux, upwind is right
                    Pp[j,n+1] = Pp[j,n] - dt_dx*( (s(x[jr])-s0)*Pp[jr,n] - (s(x[j])-s0)*Pp[j,n] ) \
                        + self.dt*(Pm[j,n] - Pp[j,n])*self.F
                elif j == self.j1: # At x1, take flux from both directions but half
                    Pp[j,n+1] = Pp[j,n] - dt_dx/2*( (s(x[jr])-s0)*Pp[jr,n] - (s(x[jl])-s0)*Pp[jl,n] ) \
                        + self.dt*(Pm[j,n] - Pp[j,n])*self.F
                        # profile num of runs
                elif j == self.j3:  # At x3, Lax-Friedrichs
                    Pp[j,n+1] = Pp[j,n] - dt_dx/2*( (s(x[jr])-s0)*Pp[jr,n] - (s(x[jl])-s0)*Pp[jl,n] ) \
                        + 1/2*(Pp[j+1,n] + Pp[j-1,n] - 2*Pp[j,n]) \
                        + self.dt*(Pm[j,n] - Pp[j,n])*self.F
                # elif j == self.j3+1:
                #     self.Pp[j,n+1] = self.Pp[j,n] - self.CFL*( (s(x[j])-s0)*self.Pp[j,n] - (s(x[idx(j-1)])-s0)*self.Pp[idx(j-1),n] ) \
                #         + self.dt*(self.Pm[j,n] - self.Pp[j,n]) \
                #         + eps*(-self.Pp[j+1,n]- self.Pp[j-1,n] + 2*self.Pp[j,n])*self.CFL
                else:
                    raise ValueError('flag')
                # Pm
                Pm[j,n+1] = Pm[j,n] - dt_dx*( (-s(x[jr])-s0)*Pm[jr,n] - (-s(x[j])-s0)*Pm[j,n] ) \
                        + self.dt*(Pp[j,n] - Pm[j,n])*self.F

            self.time_step += 1
        return None

class UpwindLFNoFlux(FiniteDiff):
    """
    Use LF flux near x3 (Source), also No flux BC left and right of x1 (Sink)
    """

    def step(self):
        """
        Advance one time step.
        """
        n = self.time_step
        s = self.s
        x = self.x_vec
        idx = self.idx
        s0 = self.s0
        dt_dx = self.dt / self.dx
        Pp = self.Pp
        Pm = self.Pm
        if self.time_step < (self.Nt-1):
            n = self.time_step
            # spatial loop
            for j in range(0, self.N):
                jl = idx(j-1)
                jr = idx(j+1)
                # Pp
                if j < self.j1l or j > self.j3:  # Positive flux, upwind is left
                    Pp[j,n+1] = Pp[j,n] - dt_dx*( (s(x[j])-s0)*Pp[j,n] - (s(x[jl])-s0)*Pp[jl,n] ) \
                        + self.dt*(Pm[j,n] - Pp[j,n])*self.F
                elif j > self.j1r and j < self.j3:  # Negative flux, upwind is right
                    Pp[j,n+1] = Pp[j,n] - dt_dx*( (s(x[jr])-s0)*Pp[jr,n] - (s(x[j])-s0)*Pp[j,n] ) \
                        + self.dt*(Pm[j,n] - Pp[j,n])*self.F
                elif j == self.j1l:  # Left of x1, No flux at j1l
                    Pp[j,n+1] = Pp[j,n] - dt_dx*(0 - (s(x[jl])-s0)*Pp[jl,n] )\
                        + self.dt*(Pm[j,n] - Pp[j,n])*self.F
                elif j == self.j1r:  # Right of x1, No flux at j1r
                    Pp[j,n+1] = Pp[j,n] - dt_dx*( (s(x[jr])-s0)*Pp[jr,n] - 0)\
                        + self.dt*(Pm[j,n] - Pp[j,n])*self.F
                elif j == self.j3:  # At x3, Lax-Friedrichs
                    Pp[j,n+1] = Pp[j,n] - dt_dx/2*( (s(x[jr])-s0)*Pp[jr,n] - (s(x[jl])-s0)*Pp[jl,n] ) \
                        + 1/2*(Pp[j+1,n] + Pp[j-1,n] - 2*Pp[j,n]) \
                        + self.dt*(Pm[j,n] - Pp[j,n])*self.F
                else:
                    raise ValueError('flag')
                # Pm
                Pm[j,n+1] = Pm[j,n] - dt_dx*( (-s(x[jr])-s0)*Pm[jr,n] - (-s(x[j])-s0)*Pm[j,n] ) \
                        + self.dt*(Pp[j,n] - Pm[j,n])*self.F

            self.time_step += 1
        return None

class UpwindLFLF(FiniteDiff):
    """
    Use LF flux near x3 (Source), also LF flux at x1 (Sink)
    The total prob mass does not conserve for some reason...
    """

    def step(self):
        """
        Advance one time step.
        """
        n = self.time_step
        s = self.s
        x = self.x_vec
        idx = self.idx
        s0 = self.s0
        dt_dx = self.dt / self.dx
        Pp = self.Pp
        Pm = self.Pm
        if self.time_step < (self.Nt-1):
            n = self.time_step
            # spatial loop
            for j in range(0, self.N):
                jl = idx(j-1)
                jr = idx(j+1)
                # Pp
                if j < self.j1 or j > self.j3:  # Positive flux, upwind is left
                    Pp[j,n+1] = Pp[j,n] - dt_dx*( (s(x[j])-s0)*Pp[j,n] - (s(x[jl])-s0)*Pp[jl,n] ) \
                        + self.dt*(Pm[j,n] - Pp[j,n])*self.F
                elif j > self.j1 and j < self.j3:  # Negative flux, upwind is right
                    Pp[j,n+1] = Pp[j,n] - dt_dx*( (s(x[jr])-s0)*Pp[jr,n] - (s(x[j])-s0)*Pp[j,n] ) \
                        + self.dt*(Pm[j,n] - Pp[j,n])*self.F
                elif j == self.j1:  # At x1, Lax-Friedrichs
                    Pp[j,n+1] = 0 - dt_dx/2*( (s(x[jr])-s0)*Pp[jr,n] - (s(x[jl])-s0)*Pp[jl,n] ) \
                        + 1/2*(Pp[j+1,n] + Pp[j-1,n]) \
                        + self.dt*(Pm[j,n] - Pp[j,n])*self.F
                elif j == self.j3:  # At x3, Lax-Friedrichs
                    Pp[j,n+1] = Pp[j,n] - dt_dx/2*( (s(x[jr])-s0)*Pp[jr,n] - (s(x[jl])-s0)*Pp[jl,n] ) \
                        + 1/2*(Pp[j+1,n] + Pp[j-1,n] - 2*Pp[j,n]) \
                        + self.dt*(Pm[j,n] - Pp[j,n])*self.F
                # elif j == self.j3+1:
                #     self.Pp[j,n+1] = self.Pp[j,n] - self.CFL*( (s(x[j])-s0)*self.Pp[j,n] - (s(x[idx(j-1)])-s0)*self.Pp[idx(j-1),n] ) \
                #         + self.dt*(self.Pm[j,n] - self.Pp[j,n]) \
                #         + eps*(-self.Pp[j+1,n]- self.Pp[j-1,n] + 2*self.Pp[j,n])*self.CFL
                else:
                    raise ValueError('flag')
                # Pm
                Pm[j,n+1] = Pm[j,n] - dt_dx*( (-s(x[jr])-s0)*Pm[jr,n] - (-s(x[j])-s0)*Pm[j,n] ) \
                        + self.dt*(Pp[j,n] - Pm[j,n])*self.F

            self.time_step += 1
        return None

class UpwindLFNoFluxItp(FiniteDiff):
    """
    Use LF flux near x3 (Source), also No flux BC left and right of x1 (Sink).

    The no flux BC at both sides of x1 is achievend by interpolation.
    """

    def step(self):
        """
        Advance one time step.
        """
        n = self.time_step
        s = self.s
        x = self.x_vec
        idx = self.idx
        s0 = self.s0
        dt_dx = self.dt / self.dx
        Pp = self.Pp
        Pm = self.Pm
        if self.time_step < (self.Nt-1):
            n = self.time_step
            # spatial loop
            for j in range(0, self.N):
                jl = idx(j - 1)
                jr = idx(j + 1)
                j1l, j1ll = self.j1l, idx(self.j1l - 1)
                j1r, j1rr = self.j1r, idx(self.j1r + 1)

                # Flux interpolation calculation
                mpl = (s(x[j1l])-s0) * Pp[j1l, n]  # Flux 1 stencil from x1 to the left
                mpll = (s(x[j1ll])-s0) * Pp[j1ll, n]  # Flux 2 stencils from x1 to the left
                mpr = (s(x[j1r])-s0) * Pp[j1r, n]  # Flux 1 stencil from x1 to the right
                mprr = (s(x[j1rr])-s0) * Pp[j1rr, n]  # Flux 2 stencil from x1 to the right
                # Left side of ddx mp
                a_coeff_l = (mpl - mpll) / ((x[j1l]-self.x1)**2 - (x[j1ll]-self.x1)**2)
                ddx_mpl = a_coeff_l * 2 * (x[j1l]-self.x1)
                # Right side of ddx mp
                a_coeff_r = (mpr - mprr) / ((x[j1r]-self.x1)**2 - (x[j1rr]-self.x1)**2)
                ddx_mpr = a_coeff_r * 2 * (x[j1r]-self.x1)

                # Pp
                if j < self.j1l or j > self.j3:  # Positive flux, upwind is left
                    Pp[j,n+1] = Pp[j,n] - dt_dx*( (s(x[j])-s0)*Pp[j,n] - (s(x[jl])-s0)*Pp[jl,n] ) \
                        + self.dt*(Pm[j,n] - Pp[j,n])*self.F
                elif j > self.j1r and j < self.j3:  # Negative flux, upwind is right
                    Pp[j,n+1] = Pp[j,n] - dt_dx*( (s(x[jr])-s0)*Pp[jr,n] - (s(x[j])-s0)*Pp[j,n] ) \
                        + self.dt*(Pm[j,n] - Pp[j,n])*self.F
                elif j == self.j1l:  # Left of x1, interpolated flux at j1l
                    Pp[j,n+1] = Pp[j,n] - self.dt * ddx_mpl\
                        + self.dt*(Pm[j,n] - Pp[j,n])*self.F
                elif j == self.j1r:  # Right of x1, No flux at j1r
                    Pp[j,n+1] = Pp[j,n] - self.dt * ddx_mpr\
                        + self.dt*(Pm[j,n] - Pp[j,n])*self.F
                elif j == self.j3:  # At x3, Lax-Friedrichs
                    Pp[j,n+1] = Pp[j,n] - dt_dx/2*( (s(x[jr])-s0)*Pp[jr,n] - (s(x[jl])-s0)*Pp[jl,n] ) \
                        + 1/2*(Pp[j+1,n] + Pp[j-1,n] - 2*Pp[j,n]) \
                        + self.dt*(Pm[j,n] - Pp[j,n])*self.F
                else:
                    raise ValueError('flag')
                # Pm
                Pm[j,n+1] = Pm[j,n] - dt_dx*( (-s(x[jr])-s0)*Pm[jr,n] - (-s(x[j])-s0)*Pm[j,n] ) \
                        + self.dt*(Pp[j,n] - Pm[j,n])*self.F

            self.time_step += 1
        return None
