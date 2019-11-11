# this is to convert the dark matter number density field to a mass density field

import numpy as np

a = np.load("numDenFilePath")  # file path to the .npy file of the dark matter number density field 
dm_mass = 0 # value can be found in data release publication. 10^10 solar mass/h (to match the unit in the target when predicting for galaxy mass)
np.save("massDenFilePath", a*dm_mass)
