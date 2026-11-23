import numpy as np
import pandas as pd 
import scipy
from datetime import datetime
from equations import GIP

#test parameters
date_time = datetime(2018, 5, 8, 5, 0, 0)
altitudes = np.arange(60,90,1)
lat = 65.13
lon = -147.47
e_mono = 100 #keV
q_mono = 1 * 6.242e8
plot_time = 1
#instantiate equations object


def main():

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
    gip.plot_densities(2)
    gip.plot_densities(5)
    gip.plot_densities(10)
    gip.plot_densities(60)
    gip.plot_densities(100)
    gip.plot_densities(200)
    

if __name__ == "__main__":
    main()

  