# HOD (instruction and original code by Siyu He; edited and maintained by Jacky H. T. Yip)

# 1) The number of central galaxies is:
         # 1   if the halo mass is larger than Mmin
         # 0   if the halo mass is smaller than Mmin
# 2) The number of satellite galaxies follow a Poisson distribution with mean: (M/M1)^alpha

# The parameters of the HOD model are thus Mmin, M1 and alpha


# This is the algorithm:

# 1) Read all the positions and radii of the halos. For the radii I would use m200. There are other definitions like c200 and top-hat200, I would use m200.
# 2) Take one particular halo (it has a mass M)
# 3) If it has a central galaxy (its mass has to be larger than Mmin), put it in the center of the halo
# 4) Compute the mean number of satellites in the halo as (M/M1)^alpha. Given that number, draw a random number with a Poissonian distribution with that mean. That will be the number of satellites of that particular halo. Place them randomly within the virial radius of the halo
# 5) repeat 2) - 4) for all halos
# 6) Compute the power spectrum of the galaxy catalogue created
# 7) Compute some kind of chi^2 = (P(k)_illustris - P(k)_HOD)^2 (only for k in the large scales, i.e. no need to fit the spectrum in the small scales)
# 8) Find the best fit of the free parameters, Mmin, M1 and alpha, by minimizing 7)
# 9) You are done!

# You can read a few more details on the HOD (slightly different) in section 4.1 of https://arxiv.org/pdf/1311.0866.pdf


import numpy as np
import h5py
import os
import scipy.optimize as op

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import sys

alpha_min, alpha_max= 1.0, 1.1
M1_min, M1_max= 20.0, 30.0
Mmin_min, Mmin_max = 0.001, 10000

thres = 1e-3

Boxsize = 205 * 18 / 32


def ReadData(typeData):  # typeData="Halo" or "Subhalo"
	if typeData == "Halo":
		data_GroupPos = h5py.File("halos/GroupPos/TNG300-1-Dark/fof_subhalo_tab_099.Group.GroupPos.hdf5")
		data_Group_M_Crit200 = h5py.File("halos/Group_M_Crit200/TNG300-1-Dark/fof_subhalo_tab_099.Group.Group_M_Crit200.hdf5")
		data_Group_R_Crit200 = h5py.File("halos/Group_R_Crit200/TNG300-1-Dark/fof_subhalo_tab_099.Group.Group_R_Crit200.hdf5")

		Pos = data_GroupPos["Group"]["GroupPos"][:]
		M = data_Group_M_Crit200["Group"]["Group_M_Crit200"]
		r = data_Group_R_Crit200["Group"]["Group_R_Crit200"]

		del data_GroupPos
		del data_Group_M_Crit200
		del data_Group_R_Crit200

		x_ind = np.logical_and(Pos[:, 0] >= (205-Boxsize)*1000, Pos[:, 0] < 205*1000)
		y_ind = np.logical_and(Pos[:, 1] >= (205-Boxsize)*1000, Pos[:, 1] < 205*1000)
		z_ind = np.logical_and(Pos[:, 2] >= (205-Boxsize)*1000, Pos[:, 2] < 205*1000)

		r_0 = r[:] > 0  # rule out r=0
		ind = np.logical_and(np.logical_and(np.logical_and(x_ind, y_ind), z_ind), r_0)

		del x_ind
		del y_ind
		del z_ind

		Pos_info = Pos[ind]
		M_info = M[ind]
		r_info = r[ind]

		del Pos
		del M
		del r

		return (Pos_info*0.001) - (205-Boxsize), M_info, r_info*0.001  # distance in Mpc/h
	
    elif typeData == "Subhalo":
		data_SubhaloPos = h5py.File("subhalos/SubhaloPos/TNG300-1/fof_subhalo_tab_099.Subhalo.SubhaloPos.hdf5")
		data_SubhaloMassType = h5py.File("subhalos/SubhaloMassType/TNG300-1/fof_subhalo_tab_099.Subhalo.SubhaloMassType.hdf5")
		data_SubhaloFlag = h5py.File("subhalos/SubhaloFlag/TNG300-1/fof_subhalo_tab_099.Subhalo.SubhaloFlag.hdf5")

		Pos = data_SubhaloPos["Subhalo"]["SubhaloPos"][:]
		sm = data_SubhaloMassType["Subhalo"]["SubhaloMassType"][:, 4]
		flag = data_SubhaloFlag["Subhalo"]["SubhaloFlag"]

		del data_SubhaloPos
		del data_SubhaloMassType
		del data_SubhaloFlag

		x_ind = np.logical_and(Pos[:, 0] >= (205-Boxsize)*1000, Pos[:, 0] < 205*1000)
		y_ind = np.logical_and(Pos[:, 1] >= (205-Boxsize)*1000, Pos[:, 1] < 205*1000)
		z_ind = np.logical_and(Pos[:, 2] >= (205-Boxsize)*1000, Pos[:, 2] < 205*1000)

		sm_0 = sm[:] > 0
		flag_0 = flag[:] > 0
		ind = np.logical_and(np.logical_and(np.logical_and(np.logical_and(x_ind, y_ind), z_ind), sm_0), flag_0)

		del x_ind
		del y_ind
		del z_ind

		print(ind.shape)
		print(Pos.shape)

		Pos_info = Pos[ind]

		del Pos
		del sm
		del flag

		return (Pos_info*0.001) - (205-Boxsize)


def calc_Mmin(alpha, M1, M, rho_gal):  # rho_gal is density of galaxies
	i=0; max_iterations= 100;
	Mmin1=Mmin_min; Mmax1=Mmin_max;

	while (i<max_iterations):
		Mmin=0.5*(Mmin1+Mmax1)
		index = M > Mmin
		rho_mean = (np.sum(index) + np.sum((M[index]/M1)**alpha))/Boxsize**3
		if (np.absolute((rho_mean - rho_gal)/rho_gal)<thres):
			i = max_iterations
		elif (rho_mean > rho_gal):
			Mmin1 = Mmin
		else:
			Mmax1 = Mmin
		i = i + 1
	return Mmin


def HOD(Pos, M, r, alpha, M1, rho_gal):  # parameters to tune: alpha, M1; 
	# Find wether to have a central galaxy
	Mmin = calc_Mmin(alpha, M1, M, rho_gal)
	if Mmin == Mmin_min or Mmin == Mmin_max:
		return 0,0
	index = M > Mmin 
	
    # For central galaxy, compute mean number of satellites in the halo
	mean_satellites_gal = np.random.poisson(lam = (M[index]/M1)**alpha)
	
    #np.savetxt('Mass.txt', np.c_[M[index], mean_satellites_gal])
	#print (alpha, M1, Mmin, np.sum(mean_satellites_gal))
	
    # Keep only the radius now
	Pos_central, r_central = Pos[index], r[index]; del Pos,r,M,index;
	Gal_Cat = Pos_central.copy()
	
    tot = np.sum(mean_satellites_gal)
	theta = np.random.uniform(low=0,high=np.pi, size=tot)
	phi = np.random.uniform(low=0, high=np.pi*2, size=tot)
	r_rand = np.random.uniform(low=0,high=1, size=tot)
	r = np.repeat(r_central, repeats=mean_satellites_gal)
	x = np.repeat(Pos_central[:,0], repeats=mean_satellites_gal)
	y = np.repeat(Pos_central[:,1], repeats=mean_satellites_gal)
	z =  np.repeat(Pos_central[:,2], repeats=mean_satellites_gal)
	sat = [x + r*r_rand*np.sin(theta)*np.cos(phi), y + r*r_rand*np.sin(theta)*np.sin(phi), z + r*r_rand*np.cos(theta)]; del theta, phi, r_rand, r, x,y,z;
	Gal_Cat = np.concatenate((Gal_Cat, np.array(sat).T))
	
    return Gal_Cat, Mmin


def PK(Gal_Cat):  # power spectrum
	#Gal_Cat = Gal_Cat % Boxsize

	#index = (Gal_Cat[:,0] < 37.5) * (Gal_Cat[:,1] < 37.5) * (Gal_Cat[:,2] < 37.5) * (Gal_Cat[:,0] >=0) * (Gal_Cat[:,1] >= 0) * (Gal_Cat[:,2] >=0)
	#Gal_Cat = Gal_Cat[index]
	
    # Doing the number count to get the galaxy number density field
	cube, edges = np.histogramdd(Gal_Cat, bins = (np.linspace(0,Boxsize,577), np.linspace(0,Boxsize,577), np.linspace(0,Boxsize,577))); np.save("HODcube.npy", cube); print(cube.shape)
	mean_raw_cube = np.sum(cube)*1./576**3
	
    # Begin to calculate the power spectrum 
	nc = cube.shape[2] # define how many cells your box has
	delta = cube/mean_raw_cube - 1.0

	# get P(k) field: export fft of data that is only real, not complex
	delta_k = np.abs(np.fft.rfftn(delta)) 
	Pk_field =  delta_k**2
	
    # get 3d array of index integer distances to k = (0, 0, 0)
	dist = np.minimum(np.arange(nc), np.arange(nc,0,-1))
	dist_z = np.arange(nc//2+1)
	dist *= dist
	dist_z *= dist_z
	dist_3d = np.sqrt(dist[:, None, None] + dist[:, None] + dist_z)

	dist_3d  = np.ravel(dist_3d)
	Pk_field = np.ravel(Pk_field)
	
	k_bins = np.arange(nc//2+1)
	k      = 0.5*(k_bins[1:] + k_bins[:-1])*2.0*np.pi/Boxsize
	Pk     = np.histogram(dist_3d, bins=k_bins, weights=Pk_field)[0]
	Nmodes = np.histogram(dist_3d, bins=k_bins)[0]
	Pk     = (Pk/Nmodes)*(Boxsize/nc**2)**3
	
	k = k[1:];  Pk = Pk[1:];
	
    return k,Pk


def Calc_Gal_Pow(dataType):
	Gal_Cat = ReadData(dataType)
	n_bar = Gal_Cat.shape[0]*1./Boxsize**3
	k, Pk = PK(Gal_Cat)
	error = (2/(k**3*Boxsize**3))**0.5*(Pk+1/n_bar)
	#np.savetxt('Pk_'+'.txt', np.c_[k, Pk, error])
	
    return k, Pk, error, n_bar


def lnlike(theta, Pk_Gal, Pk_Gal_err, Pos, M, r, rho_gal,f):
	alpha, M1 = theta
	Gal_Cat, Mmin = HOD(Pos, M, r, alpha, M1, rho_gal)	
	if Mmin == 0:
		return -np.inf
	k, Pk_Cat  = PK(Gal_Cat)
	ii = (k <= 1).sum()
	
    # With noise
	#likelihood = np.mean((Pk_Cat-Pk_Gal)**2/Pk_Gal_err**2)
	
    # Without noise
	likelihood = np.mean(((Pk_Cat[:ii]-Pk_Gal[:ii])**2))
	
    f.write('%f %f %f %f %d\n' % (alpha, M1, Mmin, likelihood, Gal_Cat.shape[0]))
	f.flush()
	os.fsync(f.fileno())
	
    return likelihood


def get_Cat(alpha, M1):
	Pos, M, r = ReadData('Halo')
	k_Gal, Pk_Gal, Pk_Gal_err, rho_gal = Calc_Gal_Pow('Subhalo')
	
	Gal_Cat, Mmin = HOD(Pos, M, r, alpha, M1, rho_gal)
	k_Cat, Pk_Cat  = PK(Gal_Cat)
	
    print (np.mean((Pk_Gal-Pk_Cat)**2))
	
    plt.figure()
	plt.loglog(k_Cat, Pk_Cat, label = '1.017172, 25.252525')
	plt.loglog(k_Gal, Pk_Gal, label = 'TNG300')
	plt.legend()
	plt.xlabel('k (h/Mpc)')
	plt.ylabel('P(k) (Mpc/h)^3')
	plt.savefig('Pk'+'.png')


def op_chi2():
	f = open('likelihood_brute_'+'_010.txt','w')
	k, Pk_Gal, Pk_Gal_err, rho_gal = Calc_Gal_Pow('Subhalo')
	Pos, M, r = ReadData('Halo')
	nll = lambda *args: lnlike(*args)
	#result = op.minimize(nll, [1., 1., 1.], args=(Pk_Gal, Pk_Gal_err, Pos, M, r), bounds = ((0.5,10),(0.1,100),(0.1,100)))
	result = op.brute(nll, args=(Pk_Gal, Pk_Gal_err, Pos, M, r, rho_gal,f), Ns=100, ranges=((alpha_min,alpha_max),(M1_min,M1_max)))
	f.close()
	
    print(result)

if __name__ == '__main__':
	#print ("begin to optimize")
	#op_chi2()
	#op_mcmc(subbox)
	get_Cat(1.017172, 25.252525)
	#for ii in range (0,8):
	#	ReadData('Halo')
	#	Calc_Gal_Pow(ii,'Subhalo')
