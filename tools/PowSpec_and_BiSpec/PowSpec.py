import numpy as np
import matplotlib.pyplot as plt


def power_spectrum_np(cube, mean_raw_cube, SubBoxSize):
    nc = cube.shape[2]  # 1024

    delta = cube / mean_raw_cube - 1.0  # overdensity

    # get P(k) field: fft of data that is only real, not complex
    delta_k = np.abs(np.fft.rfftn(delta))
    Pk_field = delta_k ** 2

    # get 3d array of index integer distances to k = (0, 0, 0)
    dist = np.minimum(np.arange(nc), np.arange(nc, 0, -1))
    dist_z = np.arange(nc // 2 + 1)
    dist *= dist
    dist_z *= dist_z
    dist_3d = np.sqrt(dist[:, None, None] + dist[:, None] + dist_z)

    dist_3d = np.ravel(dist_3d)
    Pk_field = np.ravel(Pk_field)

    k_bins = np.arange(nc // 2 + 1)
    k = 0.5 * (k_bins[1:] + k_bins[:-1]) * 2.0 * np.pi / SubBoxSize

    Pk = np.histogram(dist_3d, bins=k_bins, weights=Pk_field)[0]
    Nmodes = np.histogram(dist_3d, bins=k_bins)[0]
    Pk = (Pk / Nmodes) * (SubBoxSize / nc ** 2) ** 3

    k = k[1:];
    Pk = Pk[1:]

    return k, Pk


sim = "TNG300"
phase = 1

prediction1_name = "HODcube"
prediction2_name = "p=0_lw=500_s=13579_e=27_p=1_lw=1d05_s=12345_e=36_ng"
prediction3_name = "p=0_lw=500_s=13579_e=27_p=1_lw=2d2_s=12345_e=45_rounded_ng"

if phase == 1:
    target = np.load("Data/"+sim+"/StellarSubhalos_Data_StarsMassesNum/subhalos_"+sim+"-1_flagged_StarsMassesNum_grid.npy")
elif phase == 2:
    target = np.load("Data/"+sim+"/StellarSubhalos_Data_StarsMasses/subhalos_"+sim+"-1_flagged_StarsMasses_grid.npy")

predictions1 = np.load(sim+"/npyFolder/"+prediction1_name+".npy")
predictions2 = np.load(sim+"/npyFolder/"+prediction2_name+".npy")
predictions3 = np.load(sim+"/npyFolder/"+prediction3_name+".npy")

tar = target[448:, 448:, 448:]
pred1 = predictions1
pred2 = predictions2
pred3 = predictions3

p1_k, p1_pk0 = power_spectrum_np(pred1, pred1.mean(), 115.3125)
p2_k, p2_pk0 = power_spectrum_np(pred2, pred2.mean(), 115.3125)
p3_k, p3_pk0 = power_spectrum_np(pred3, pred3.mean(), 115.3125)
t_k, t_pk0 = power_spectrum_np(tar, tar.mean(), 115.3125)

grid = plt.GridSpec(3, 1)

plt.figure(figsize=[9,9])
plt.rc("font", size=16)

plt.rc('axes', titlesize=22)
plt.rc('axes', labelsize=22)
plt.rc('legend', fontsize=17)

plt.subplot(grid[0:2, 0])

plt.plot(p2_k,p2_pk0,label="Cascade model", linewidth=8.5, color="#F338FC", alpha=0.8)
#plt.plot(p3_k,p3_pk0,label="CNN cascade w/ rounding", linewidth=2.5, linestyle="dashed", color="#F22C2C")
plt.plot(p1_k,p1_pk0,label="HOD",linewidth=5.2, linestyle="dotted", color="#0BB50B")
plt.plot(t_k,t_pk0,label="Target (TNG300-1)", linewidth=3.6, color="#2F00B8")
plt.axhline(y=(115.3125**3)/tar.sum(), label="Reference shot noise", color='#A399A3', linestyle="dashdot", linewidth=2.2) 

plt.xscale('log')
plt.yscale('log')
plt.xlabel('k [h/Mpc]')
plt.ylabel('P(k) [(Mpc/h)$^3$]')
#plt.title("")
plt.legend()

plt.subplot(grid[2, 0])

plt.plot(p2_k, (p2_pk0/t_pk0)**1, label="Cascade model", linewidth=5.2, color="#F338FC", alpha=0.8)
#plt.plot(p3_k, (p3_pk0/t_pk0)**1, label="CNN cascade w/ rounding",linewidth=2.5, linestyle="dashed", color="#F22C2C")
plt.plot(p1_k, (p1_pk0/t_pk0)**1, label="HOD", linewidth=3.2,linestyle="dotted", color="#0BB50B")
plt.plot(t_k, (t_pk0/t_pk0)**1, label="Target (TNG300-1)", linewidth=2.2,color="#2F00B8")

plt.xscale("log")
plt.xlabel('k [h/Mpc]')
plt.ylabel('T(k)') 
#plt.legend()

plt.savefig(sim+"/npyFolder/"+"PS_NR_NT.png")
plt.close()

MRR_1 = (np.abs(p2_pk0 - t_pk0)/np.abs(t_pk0)).mean()*100
MRR_HOD = (np.abs(p1_pk0 - t_pk0)/np.abs(t_pk0)).mean()*100

print(MRR_1)
print(MRR_HOD)