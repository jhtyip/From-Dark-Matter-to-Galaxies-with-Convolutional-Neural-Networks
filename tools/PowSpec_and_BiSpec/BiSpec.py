import numpy as np
#import Pk_library.bispectrum_library as PKL_b
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

ii = 2.5*np.pi/180.0
theta_range = np.arange(0.0*np.pi/180.0, 180.0*np.pi/180.0-ii, ii).astype("float32")

sim = "TNG300"
'''
cube_T = np.load("Data/"+sim+"/StellarSubhalos_Data_StarsMassesNum/subhalos_"+sim+"-1_flagged_StarsMassesNum_grid.npy")[448:, 448:, 448:]
delta_T = cube_T/cube_T.mean() - 1
Bk_T = PKL_b.Bk(delta_T.astype("float32"), 115.3125, 1.2, 1.3, theta_range)
np.save(sim+"/npyFolder/Bk_T.npy", Bk_T.B)



cube_1 = np.load(sim+"/npyFolder/p=0_lw=500_s=13579_e=27_p=1_lw=1d05_s=12345_e=36_ng.npy")
delta_1 = cube_1/cube_1.mean() - 1
Bk_1 = PKL_b.Bk(delta_1.astype("float32"), 115.3125, 1.2, 1.3, theta_range)
np.save(sim+"/npyFolder/Bk_1.npy", Bk_1.B)

cube_1_rounded = np.load(sim+"/npyFolder/p=0_lw=500_s=13579_e=27_p=1_lw=2d2_s=12345_e=45_rounded_ng.npy")
delta_1_rounded = cube_1_rounded/cube_1_rounded.mean() - 1
Bk_1_rounded = PKL_b.Bk(delta_1_rounded.astype("float32"), 115.3125, 1.2, 1.3, theta_range)
np.save(sim+"/npyFolder/Bk_1_rounded.npy", Bk_1_rounded.B)

cube_HOD = np.load(sim+"/npyFolder/HODcube.npy")
delta_HOD = cube_HOD/cube_HOD.mean() - 1
Bk_HOD = PKL_b.Bk(delta_HOD.astype("float32"), 115.3125, 1.2, 1.3, theta_range)
np.save(sim+"/npyFolder/Bk_HOD.npy", Bk_HOD.B)
'''


Bk_T = np.load(sim+"/npyFolder/Bk_T.npy")
Bk_1 = np.load(sim+"/npyFolder/Bk_1.npy")
Bk_1_rounded = np.load(sim+"/npyFolder/Bk_1_rounded.npy")
Bk_HOD = np.load(sim+"/npyFolder/Bk_HOD.npy")


theta_range = theta_range/np.pi
grid = plt.GridSpec(3, 1)


plt.figure(figsize=[9,9])
plt.rc("font", size=16)

plt.rc('axes', titlesize=22)
plt.rc('axes', labelsize=22)
plt.rc('legend', fontsize=17)

plt.subplot(grid[0:2, 0])

plt.plot(theta_range, Bk_1/1000000, label="Cascade model", linewidth=8.5, color="#F338FC", alpha=0.7)
#plt.plot(theta_range, Bk_1_rounded/1000000, label="CNN cascade w/ rounding",linewidth=2.5, linestyle="dashed", color="#F22C2C")
plt.plot(theta_range, Bk_HOD/1000000, label="HOD", linewidth=5.2,linestyle="dotted", color="#0BB50B")
plt.plot(theta_range, Bk_T/1000000, label="Target (TNG300-1)", linewidth=3.2,color="#2F00B8")


plt.xlabel(r"$\Theta$/$\pi$")
plt.ylabel("B(k$_1$, k$_2$, $\Theta$) [10$^6$ (Mpc/h)$^6$]")
plt.title("k$_1$ = 1.2 h/Mpc, k$_2$ = 1.3 h/Mpc")
plt.legend()
'''
plt.subplot(grid[2, 0])

plt.plot(theta_range, Bk_1/Bk_T, label="Cascade model", linewidth=6.5, color="#F338FC", alpha=0.7)
#plt.plot(theta_range, Bk_1_rounded/Bk_T, label="CNN cascade w/ rounding",linewidth=2.5, linestyle="dashed", color="#F22C2C")
plt.plot(theta_range, Bk_HOD/Bk_T, label="HOD", linewidth=3.2,linestyle="dotted", color="#0BB50B")
plt.plot(theta_range, Bk_T/Bk_T, label="Target (TNG300-1)", linewidth=2.2,color="#2F00B8")


plt.xlabel(r"$\Theta$/$\pi$")
plt.ylabel("T(k$_1$, k$_2$, $\Theta$)")
plt.yticks(np.arange(1, 9, 2))
'''
plt.savefig(sim+"/npyFolder/"+"BS_NR_NT.png")
plt.close()

MRR_1 = (np.abs(Bk_1 - Bk_T)/np.abs(Bk_T)).mean()*100
MRR_HOD = (np.abs(Bk_HOD - Bk_T)/np.abs(Bk_T)).mean()*100

print(MRR_1)
print(MRR_HOD)

