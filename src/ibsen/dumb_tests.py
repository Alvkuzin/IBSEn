import numpy as np
import matplotlib.pyplot as plt


a = ( [1, 2] , [300, 500] , [1000, 2000] )

E_ = []
for band in a:
    N_in_band = int(30 * (np.log10(band[1]) - np.log10(band[0])))
    E_in_band = np.logspace(np.log10(band[0]), np.log10(band[1]), N_in_band)
    E_.append(E_in_band)
E_ = np.concatenate(E_)
                   
plt.plot(E_, np.ones_like(E_), 'o')
plt.xscale('log')
plt.show()