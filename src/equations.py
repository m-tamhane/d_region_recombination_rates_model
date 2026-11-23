import numpy as np
import pandas as pd 
import scipy
from datetime import datetime
from nrlmsise00 import msise_flat
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import PyIRI
import PyIRI.main_library as ml
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D



#Things to improve--the molar weight DOES matter per altitude and makes rapid changed compared the the approximated version. If going beyond 90 KM, this must be changed so that the code looks up the proper molecular weights for various 




class GIP:
    def __init__(self):

        #constant coefficients needed for electron transport model
        self.P = np.array(
        [
            [ 1.24616e+0,  1.45903e+0, -2.42269e-1,  5.95459e-2],  # i = 1
            [ 2.23976e+0, -4.22918e-7,  1.36458e-2,  2.53332e-3],  # i = 2
            [ 1.41754e+0,  1.44597e-1,  1.70433e-2,  6.39717e-4],  # i = 3
            [ 2.48775e-1, -1.50890e-1,  6.30894e-9,  1.23707e-3],  # i = 4
            [-4.65119e-1, -1.05081e-1, -8.95701e-2,  1.22450e-2],  # i = 5
            [ 3.86019e-1,  1.75430e-3, -7.42960e-4,  4.60881e-4],  # i = 6
            [-6.45454e-1,  8.49555e-4, -4.28581e-2, -2.99302e-3],  # i = 7
            [ 9.48930e-1,  1.97385e-1, -2.50660e-3, -2.06938e-3],  # i = 8
        ]) 
        
        self.dt = 0.1 # set to 1ms for now, should create a function later
        self.t_end = 200
        self.t_array = np.arange(0,self.t_end + self.dt, self.dt)


    #set parameters for 5 stiff ODEs
    def set_msise_parameters(self,date_time: datetime,altitudes: np.ndarray, lat,lon, e_mono:np.ndarray):
        self.date_time = date_time
        self.altitudes = altitudes
        self.lat = lat
        self.lon = lon
        self.e_mono = e_mono # needs to be in keV !!!
        self.get_background_density()
        return
    def set_recombination_rates(self):
        self.neutrals_number_density, self.neutral_temperature, self.neutrals_mass = self.get_neutrals(self.date_time,self.altitudes,self.lat,self.lon) #neutral mass density in g /cm^3 and temp in K ofc
        self.a_c_d = 1e-5 #cm^3 / s^-1
        self.a_i = 1e-7 + (1e-24 * (self.neutrals_number_density)) #ion-ion recombination between positive and negative ions
        self.a_d = 3e-7 #efffective coefficient of dissociative recombination, i'm keeping this constant because i'm only modeling the d region
        self.beta = ((self.neutrals_number_density)**2) * (5.6e-32)
        self.B = ((self.neutrals_number_density)**2)* 1e-31 #conversion rate from light positive ions to heavy cluster ions
        self.gamma_x = 0 #VLF sferic matching pipeline only works well at night!
        self.x_bar = (0.2e-30) * ((self.neutrals_number_density)**2) #conversion rate from light to heavy negative ions
        self.gamma = (8.61e-10)*self.neutrals_number_density*np.exp(-6030/self.neutral_temperature)

        self.rates_df = pd.DataFrame(
        {
        'gamma':    self.gamma.ravel(),
        'gamma_x':  self.gamma_x,
        'beta':     self.beta.ravel(),
        'alpha_d':  self.a_d,
        'alpha_dc': self.a_c_d,
        'alpha_i':  self.a_i.ravel(),
        'B':        self.B.ravel(),
        'X_bar':    self.x_bar.ravel(),
        },
        index=self.altitudes
        )
        self.rates_df.index.name = 'altitude_km'

    def ode_main(self):
        '''
        Main Wrapper function for Solver

        returns:
            background(ndarray): (altitudes,densities,time)
            solution(ndarray): (altitudes,densities,time)
        '''
        #sets needed object parameters for ODEs, recombination rates and q_total
        self.set_recombination_rates()
        print('Recombination Rates Set')
        #retrieve ionization impact of electron precipitation
        self.ionization_model_init()
        print('Ionization Model Initialized')

        background = np.zeros((len(self.altitudes),5))
        solutions = np.zeros((len(self.altitudes),5,len(self.t_array)))
        #iterate over the altitudes and calculate the solutions

        print('Solving ODEs')
        for idx,altitude in enumerate(self.altitudes):
            print(f'Solving ODEs at {altitude}')
            background_densities, solved_densities = self.ode_solve(altitude)
            background[idx,:] = background_densities
            solutions[idx,:,:] = solved_densities
        self.background = background
        self.solutions = solutions
        return background, solutions

    def ode_solve(self,altitude:float):

        #ne background from IRI
        ne_background = self.IRI.loc[altitude].to_numpy()


        #find initial value/condition
        initial_condition = np.array([ne_background[0],0,0,ne_background[0],0])
        q0, background_densities = self.bounded_bisection(
            background_electron_density = ne_background,
            initial_condition=initial_condition,
            altitude = altitude, 
            itermax = 100)

        print(f'Solved Background q0 at {altitude}')
        q = q0 + self.q_df.loc[altitude].values.item()
        #now that we have an initial guess for q, we can solve the non linear ODEs using Euler Backward
        solved_densities = self.euler_backward(initial_condition = background_densities[:,-1],q = q, altitude = altitude)
        return background_densities[:,-1], solved_densities
    def bounded_bisection(self,background_electron_density:float, initial_condition:np.ndarray, altitude:float, itermax: float, error_bound = 0.01):
        '''
        Solves an inverse problem: we require the ionoization rate at the background electron density. 
        While the background electron density is known, the ionization rate is not so we need to 
        do the bisection method until the electron density matches the background
        args: 
            background_electron_density(float): an IRI provided value (or measured value)
            initial_condition(np.ndarray): dummy condition to start bisection method
            itermax(float):number of iterations before method quits
            error_bound(float): difference between computed electron density and background electron density required before convergence

        '''
        q_low = 1e-6
        q_high = 1e6 #i got these bounds from the profiles in Kaeppler et al
        #one iteration of the euler backward
        #jacobian implicit method needed here
        q_init = np.sqrt(q_low*q_high)
        densities = self.euler_backward(initial_condition,q_init,altitude)
        computed_electron_density = densities[0,-1]
        error = computed_electron_density - background_electron_density
        rel_tol = (error_bound*background_electron_density)
        iter = 0
        while((abs(error) > rel_tol ) and iter < itermax):
            if (error > 0): #need to go low
                q_high = q_init
                q_init = np.sqrt((q_init*q_low))
            elif (error < 0): #need to go high
                q_low = q_init
                q_init = np.sqrt((q_init*q_high))
            densities = self.euler_backward(initial_condition,q_init,altitude)
            computed_electron_density = densities[0,-1]
            error = (computed_electron_density - background_electron_density)
            
            iter += 1


        return q_init, densities

            
    def euler_backward(self, initial_condition:np.ndarray,q,altitude):
        '''
        Euler backward solver for non linear system using newton raphson

        args:
            initial_condition(ndarray) : N_0 of each density for 5 odes
            q(float): ionization rate
            altitude(float): altitude in km to solve for
        return
            out(ndarray): (5,len(t_array)) --densities at each time step
        '''
        out = np.zeros((5,len(self.t_array)))
        out[:,0] = initial_condition
        for idx in range(len(self.t_array) - 1 ):
            out[:,idx+1] = self.newton_raphson(out[:,idx],q,altitude)
        return out
    def plot_densities(self, time: float):
        fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

        # Safer than exact float equality (==) when searching self.t_array
        index = int(np.argmin(np.abs(self.t_array - time)))

        bg_densities = self.background[:, :]   # shape: (N_alt, 5)
        sol_densities = self.solutions[:, :, index]   # shape: (N_alt, 5)
        species_labels = [r'$n_e$', r'$n^-$', r'$n_x^-$', r'$n^+$', r'$n_x^+$']
        colors = plt.cm.tab10(np.arange(5))

        def make_line_collection(densities_2d):
            # Build segments of shape (5, N_alt, 2) where each point is (density, altitude)
            segments = np.stack(
                [np.column_stack([densities_2d[:, i], self.altitudes]) for i in range(5)],
                axis=0
            )
            return LineCollection(segments, colors=colors, linewidths=2.0)

        # Left panel: Background densities
        lc_bg = make_line_collection(bg_densities)
        axs[0].add_collection(lc_bg)
        axs[0].autoscale()  # Required because add_collection() does not auto-update axis limits
        axs[0].set_xscale('log')
        axs[0].set_xlabel(r"Density ($\mathrm{cm}^{-3}$)")
        axs[0].set_ylabel("Altitude (km)")
        axs[0].set_title(f"Background Densities (t = {self.t_array[index]:.2f} s)")

        # Right panel: Post-precipitation densities
        lc_sol = make_line_collection(sol_densities)
        axs[1].add_collection(lc_sol)
        axs[1].autoscale()
        axs[1].set_xscale('log')
        axs[1].set_xlabel(r"Density ($\mathrm{cm}^{-3}$)")
        axs[1].set_title(f"Densities after {self.e_mono} keV (t = {self.t_array[index]:.2f} s)")

        # Shared legend using proxy artists (LineCollection contains 5 lines in one artist)
        legend_handles = [
            Line2D([0], [0], color=colors[i], lw=2.0, label=species_labels[i])
            for i in range(5)
        ]
        axs[1].legend(handles=legend_handles, loc="best")

        fig.tight_layout()
        plot_save_dir = Path.cwd().parent / "plots"
        plot_save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_save_dir / f"electron_density_Qo_1_{self.t_array[index]:.2f}.png", dpi=150)

        
    def newton_raphson(self, initial_condition:np.ndarray,q:float,altitude:float):
        '''
        args:
            initial_condition(ndarray) : (5,) array with the initial conditions for electron and ion densities
            q(float): ionization rate 
            altitude(float) : (1,) altitude to pull recombination rates for
        return
            y_n_v_1(np.ndarray): (5,) next iteration for each value
        '''
        y_0 = initial_condition
        err_tol = 0.01 # 1% error tolerance
        #first iteration
        y_n_v = y_0.copy()
        I = np.eye(5)
        for _ in range(50):  # Always include a max-iteration guard!
            R = y_n_v - y_0 - self.dt * self.odes(y_n_v, q, altitude)
            M = I - self.dt * self.jacobian(y_n_v, altitude)
            delta_y = np.linalg.solve(M, -R)
            y_n_v_1 = y_n_v + delta_y
            if np.all(np.abs(delta_y) <= err_tol * np.abs(y_n_v) + 1e-10):
                break
            y_n_v = y_n_v_1
        return y_n_v_1

    def odes(self,N:np.ndarray, q:float, altitude:float):
        """
        Computes the time derivatives (dy/dt) for the 5-fluid D- and E-region 
        ion chemistry model (Equations 1a-1e from Kaeppler et al., 2022 / Lehtinen & Inan, 2007).
        
        Parameters
        ----------
        N : array-like of shape (5,)
            State vector of densities [n_e, n_minus, n_x_minus, n_plus, n_x_plus] in cm^-3.
        q : float
            Altitude ionization rate in cm^-3 s^-1.
        altitude : float
            altitude where the ODEs are solved in km
        Returns
        -------
        dydt : ndarray of shape (5,)
            Time derivatives [dn_e/dt, dn-/dt, dnx-/dt, dn+/dt, dnx+/dt] in cm^-3 s^-1.
        """
        n_e, n_m, n_xm, n_p, n_xp = N
        total_pos = n_p + n_xp
        total_neg = n_m + n_xm

        #pull rates from rates_df(pandas dataframe indexed by altitude)
        rates = self.rates_df.loc[altitude]
        gamma = rates['gamma']
        gamma_x = rates['gamma_x']
        beta = rates['beta']
        X_bar = rates['X_bar']
        alpha_d = rates['alpha_d']
        alpha_i = rates['alpha_i']
        alpha_dc = rates['alpha_dc']
        B = rates['B']

        dne_dt  = q + gamma * n_m + gamma_x * n_xm - beta * n_e - alpha_d * n_e * n_p - alpha_dc * n_e * n_xp
        dnm_dt  = beta * n_e - gamma * n_m - alpha_i * n_m * total_pos - X_bar * n_m
        dnxm_dt = -gamma_x * n_xm - alpha_i * n_xm * total_pos + X_bar * n_m
        dnp_dt  = q - B * n_p - alpha_d * n_e * n_p - alpha_i * total_neg * n_p
        dnxp_dt = -alpha_dc * n_e * n_xp + B * n_p - alpha_i * total_neg * n_xp
        
        return np.array([dne_dt, dnm_dt, dnxm_dt, dnp_dt, dnxp_dt], dtype=float)
        
    def jacobian(self,N:np.ndarray,altitude:float):
        """
        Computes the exact 5x5 analytical Jacobian matrix J = df/dy for the 
        GPI5 ODE system.

        Parameters
        ----------
        N : array-like of shape (5,)
            State vector of densities [n_e, n_minus, n_x_minus, n_plus, n_x_plus] in cm^-3.
        altitude : float
            altitude for which to calculate jacobian
            
        Returns
        -------
        J : ndarray of shape (5, 5)
            The 5x5 Jacobian matrix where J[i, j] = df_i / dy_j.
        """
        n_e, n_m, n_xm, n_p, n_xp = N
        rates = self.rates_df.loc[altitude]
        gamma = rates['gamma']
        gamma_x = rates['gamma_x']
        beta = rates['beta']
        X_bar = rates['X_bar']
        alpha_d = rates['alpha_d']
        alpha_i = rates['alpha_i']
        alpha_dc = rates['alpha_dc']
        B = rates['B']

        total_pos = n_p + n_xp
        total_neg = n_m + n_xm
        J = np.zeros((5,5), dtype=float)
        # Row 0: d(dn_e/dt) / dy
        J[0, 0] = -beta - alpha_d * n_p - alpha_dc * n_xp
        J[0, 1] = gamma
        J[0, 2] = gamma_x
        J[0, 3] = -alpha_d * n_e
        J[0, 4] = -alpha_dc * n_e
        
        # Row 1: d(dn-/dt) / dy
        J[1, 0] = beta
        J[1, 1] = -gamma - X_bar - alpha_i * total_pos
        J[1, 2] = 0.0
        J[1, 3] = -alpha_i * n_m
        J[1, 4] = -alpha_i * n_m
        
        # Row 2: d(dn_x-/dt) / dy
        J[2, 0] = 0.0
        J[2, 1] = X_bar
        J[2, 2] = -gamma_x - alpha_i * total_pos
        J[2, 3] = -alpha_i * n_xm
        J[2, 4] = -alpha_i * n_xm
        
        # Row 3: d(dn+/dt) / dy
        J[3, 0] = -alpha_d * n_p
        J[3, 1] = -alpha_i * n_p
        J[3, 2] = -alpha_i * n_p
        J[3, 3] = -B - alpha_d * n_e - alpha_i * total_neg
        J[3, 4] = 0.0
        
        # Row 4: d(dn_x+/dt) / dy
        J[4, 0] = -alpha_dc * n_xp
        J[4, 1] = -alpha_i * n_xp
        J[4, 2] = -alpha_i * n_xp
        J[4, 3] = B
        J[4, 4] = -alpha_dc * n_e - alpha_i * total_neg
        
        return J








    #build ionization model derived from "Parameterization of monoenergetic electron impact ionization" (Fan et al 2010)
    def set_q_mono(self,q_mono):
        self.q_mono = q_mono
        return
    def calculate_y(self):
        '''
        args: self.neutrals_mass, self.altitudes, described above
        returns y(np.ndarray): normalized atmospheric column mass as function of vertical location
        '''
        kb_cgs = 1.380649e-23 #kg m^2
        R = 8.3145 #J/mol K
        Re = 6371.00877 #in km
        M0 = 28.964*np.ones([len(self.altitudes),1]) #vert vector of molar weight, in
        
        Na = 6.02214076e23 
        M0[len(self.altitudes)-1] = 28.91
        M0[len(self.altitudes)-2] = 28.93 #I found the more 'exact' values from the ISA (International Standard Atmosphere)
        M0[len(self.altitudes)-3] = 28.94
        M0[len(self.altitudes)-4] = 28.95
        M0 = M0*1e-3
        g = np.atleast_2d(9.806665*(Re / (Re + self.altitudes))**2)
        self.scale_height = (R * self.neutral_temperature.T) / (M0 *g.T) *1e2 #scale height calculated in m, need cm per fang et al
        y = (2/self.e_mono) * (((self.neutrals_mass.T) * (self.scale_height)) / 6e-6)**(0.7)


        return y
    def calculate_coefficients(self):
        '''
        args:
            self.e_mono(np.ndarray) : an array of the electron energy levels we will create an ionization model for (using Fang et al 2010)
        returns:
            C(np.ndarray) : array of coefficients for calculating ionoization model
        '''
        #type check e_mono for single scalar value
        if type(self.e_mono) != np.ndarray:
            self.e_mono = np.array([self.e_mono])
        #make j array for broadcasting
        j = np.arange(0,4,1)
        #calculate natlog for every j
        nat_log = (np.log(self.e_mono[:,np.newaxis]))**(j)
        #calculate coefficient table for given electron energy levels
        self.C = np.exp(np.sum(((self.P.T.reshape(1,4,8))*nat_log[:,:,np.newaxis]),axis = 1))


        return self.C
    def calculate_energy_dissipation(self, C:np.ndarray, y: np.ndarray):
        c1 = C[:,0]
        c2 = C[:,1]
        c3 = C[:,2]
        c4 = C[:,3]
        c5 = C[:,4]
        c6 = C[:,5]
        c7 = C[:,6]
        c8 = C[:,7]
        f = c1*(y**c2)*np.exp(-1*c3*(y**(c4)))+c5*(y**(c6))*np.exp(-1*c7*(y**(c8)))


        return f
    def calculate_q_total(self):
        self.C = self.calculate_coefficients()
        self.y = self.calculate_y()
        self.f = self.calculate_energy_dissipation(self.C,self.y)
        
        normalization_factor = (self.q_mono / (2*0.035) )* (1/ self.scale_height)
        q_total = (self.f * normalization_factor)  
        return q_total 
    #method for defining ionization model
    def ionization_model_init(self):
        if not hasattr(self, "q_mono"):
            self.q_mono = 1 # erg cm^-2 s _1

        if not hasattr(self,("neutrals_mass")):
            print("Atmospheric Neutral Mass Density not found. " \
            "Use class method get_neutrals to load data from NRLMSISE Model." \
            "Exiting")
            sys.exit(-1)
            return
        self.q_total = self.calculate_q_total()

        self.q_df = pd.DataFrame(
        {
            'q': self.q_total.ravel()
        },
        index = self.altitudes
        )

        return 
    
    #plotting functions
    def plot_ionization_model(self):

        fig, ax = plt.subplots(1)
        
        ax.set_title(f'Ionization over 60 - 90 km (D-region) at {self.e_mono} (keV)')
        ax.set_xlabel(r"Ionization ($\mathrm{cm}^{-3}{s}^{-1}$)")
        ax.set_ylabel("Altitude (km)")
        ax.plot(self.q_total,self.altitudes)
        plot_save_dir = Path.cwd().parent / "plots"
        plot_save_dir.mkdir(parents = True, exist_ok = True)
        plt.savefig(plot_save_dir / "ionization_test_Qo_1")
    def plot_normalized_energy_deposition(self):


        fig, ax = plt.subplots(1)
        
        ax.set_title(f'Normalized Energy Depositions vs Normalized Atmospheric Column Mass')
        # ax.set_xscale("log")
        ax.set_xlabel("Normalized Energy Deposition")
        ax.set_ylabel("Normalized Atmospheric Column Mass")
        ax.plot(self.f,self.y)
        plot_save_dir = Path.cwd().parent / "plots"
        plot_save_dir.mkdir(parents = True, exist_ok = True)
        plt.savefig(plot_save_dir / "energy_deposition_test_Qo_1")
    def plot_recombination_rates(self):

        fig,axs = plt.subplots(1,2)
        axs[0].scatter(self.a_i,self.altitudes, label = 'a_i', color = 'black')
        axs[0].axvline(self.a_c_d,label = 'a_c_d', color = 'orange')
        axs[0].axvline(self.a_d,label = 'a_d', color = 'blue')
        axs[1].scatter(self.beta,self.altitudes, label = 'beta')
        axs[1].scatter(self.B,self.altitudes, label = 'B' )
        axs[1].scatter(self.gamma,self.altitudes, label = 'gamma')
        axs[1].axvline(self.gamma_x, label = 'gamma_x')
        axs[1].scatter(self.x_bar,self.altitudes, label = 'x_bar')
        axs[0].set_xlabel(r"($cm^{3} s^{-1}$)")
        axs[0].set_xscale("log")
        axs[1].set_xscale("log")
        axs[1].set_xlabel(r"($s^{-1}$)")
        axs[0].set_ylabel("Altitude (km)")
        axs[0].set_title("Recombination Rates")
        axs[0].legend()
        axs[1].legend()


        plot_save_dir = Path.cwd().parent / "plots"
        plot_save_dir.mkdir(parents = True, exist_ok = True)
        plt.savefig(plot_save_dir / "recombination_rates_test_Qo_1")


    #method for defining ODEs

   

    @staticmethod 
    def get_neutrals(date_time: datetime, altitudes: np.ndarray, lat,lon):
        '''
        args:
            datetime(datetime.datetime): the date and time for obtaining neutral gas density measurements
            altitudes(np.ndarray): the altitude in the atmosphere we want the density measurements for
            lat(float): latitude 
            lon(float): longitude

        returns: (np.ndarray) total mass density of neutrals at given altitudes and coordinate
                 (np.ndarray) temperature at altitude     
        '''
        Na = 6.02214076e23 # avogadros numer
        M0 = 28.964
        #import navy research NRLMSISE-00 via python wrapper for C code
        neutral_density =np.atleast_2d(np.array(msise_flat(date_time,altitudes,lat,lon,150,150,4)))

        neutral_number_density = np.atleast_2d(neutral_density[:,5] * Na / M0 )
        neutral_temperature = np.atleast_2d(neutral_density[:,10])
        neutral_mass_density = np.atleast_2d(neutral_density[:,5])
        return neutral_number_density, neutral_temperature, neutral_mass_density  #number density,temperature, mass density for each altitude

    def get_background_density(self):
        year = self.date_time.year
        month = self.date_time.month
        day = self.date_time.day
        aUT = np.array([self.date_time.hour])
        alt_res = 1
        F107 = 100
        ccir_or_ursi = 0  # 0 = CCIR, 1 = URSI
        _,_,_,_,_,_,background_density_profile = ml.IRI_density_1day(year,month,day,aUT,np.array([self.lat]),np.array([self.lon]),self.altitudes, F107, PyIRI.coeff_dir, ccir_or_ursi )
        # breakpoint()
        background_density_profile = background_density_profile * 1e-6
        self.IRI = pd.DataFrame(
            {'background_density': background_density_profile.ravel()
            },
            index = self.altitudes
        )
def main():
        #test parameters
    date_time = datetime(2018, 5, 8, 5, 0, 0)
    altitudes = np.arange(60,90,1)
    lat = 65.13
    lon = -147.47
    e_mono = 300 #keV
    q_mono = 1 * 6.242e8
    plot_time = 1
    #instantiate equations object
    gip = GIP()

    #set_msise_parameters
    gip.set_msise_parameters(date_time = date_time, altitudes = altitudes, lat = lat, lon = lon, e_mono = e_mono)
    #we require the recombination rates of the D-region ionosphere as well as the ionoization model
    #set incident energy flux
    gip.set_q_mono(q_mono)
    #initialization ionization_model
    gip.ode_main()

    gip.plot_normalized_energy_deposition()
    gip.plot_recombination_rates()
    gip.plot_ionization_model()
    gip.plot_densities(plot_time)


    # #Debugging Plotting Sciprts
if __name__ == "__main__":
    main()



  

