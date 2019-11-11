import numpy as np
import matplotlib.pyplot as plt
import math
from itertools import product
from matplotlib.colors import Normalize
import matplotlib.patches as patches

model_ng = "p=0_lw=500_s=13579_e=15_p=1_lw=2d25_s=12345_e=26_rounded_ng"
model_gm = "p=0_lw=500_s=13579_e=15_p=1_lw=2d25_s=12345_e=26_p=2_lw=2d3_r=True_s=22_e=28_gm"

sim = "TNG300"
#phase = 2

i = 16

if i == 1:
    sim = "TNG300"
    #ng_pre = np.load(sim+"/npyFolder/"+model_ng+".npy")
    #gm_pre = np.load(sim+"/npyFolder/"+model_gm+".npy")[512:, 512:, 512:]
    #ng_tar = np.load("Data/"+sim+"/StellarSubhalos_Data_StarsMassesNum/subhalos_"+sim+"-1_flagged_StarsMassesNum_grid.npy")[448:, 448:, 448:]
    #gm_tar = np.load("Data/"+sim+"/StellarSubhalos_Data_StarsMasses/subhalos_"+sim+"-1_StarsMasses_grid.npy")[512:, 512:, 512:]

    cube_T = np.load("Data/" + sim + "/StellarSubhalos_Data_StarsMassesNum/subhalos_" + sim + "-1_flagged_StarsMassesNum_grid.npy")[448:, 448:, 448:]
    cube_1 = np.load(sim + "/npyFolder/p=0_lw=500_s=13579_e=27_p=1_lw=1d05_s=12345_e=36_ng.npy")
    cube_1_rounded = np.load(sim + "/npyFolder/p=0_lw=500_s=13579_e=27_p=1_lw=2d2_s=12345_e=45_rounded_ng.npy")
    cube_HOD = np.load(sim + "/npyFolder/HODcube.npy")


    '''
    ng_negative_vox = np.sum(ng_pre < 0)
    gm_negative_vox = np.sum(gm_pre < 0)
    '''
    #ng_pre[ng_pre < 0] = 0
    #ng_pre = ng_pre.round()
    #gm_pre[gm_pre < 0] = 0
    '''
    ng_mse = np.sum((ng_pre - ng_tar)**2) / ng_pre.size
    gm_mse = np.sum((gm_pre - gm_tar)**2) / gm_pre.size
    
    ng_index_eitherNonEmpty = ((ng_pre > 0) + (ng_tar > 0)) > 0
    ng_mse_eitherNonEmpty = np.sum((ng_pre[ng_index_eitherNonEmpty] - ng_tar[ng_index_eitherNonEmpty])**2) / ng_pre[ng_index_eitherNonEmpty].size
    gm_index_eitherNonEmpty = ((ng_pre > 0) + (ng_tar > 0)) > 0
    gm_mse_eitherNonEmpty = np.sum((gm_pre[gm_index_eitherNonEmpty] - gm_tar[gm_index_eitherNonEmpty])**2) / gm_pre[gm_index_eitherNonEmpty].size
    
    totalVox = ng_pre.size
    ng_nonEmptyVox_pre = np.sum(ng_pre > 0)
    ng_nonEmptyVox_tar = np.sum(ng_tar > 0)
    gm_nonEmptyVox_pre = np.sum(gm_pre > 0)
    gm_nonEmptyVox_tar = np.sum(gm_tar > 0)
    
    # take ng_pre as ground truth
    index_bothNonEmpty = (gm_pre > 0) * (ng_pre > 0)
    index_bothEmpty = (gm_pre == 0) * (ng_pre == 0)
    preNonEmpty_TarNot = (gm_pre > 0) * (ng_pre == 0)
    preEmpty_TarNot = (gm_pre == 0) * (ng_pre > 0)
    
    TP = np.sum(index_bothNonEmpty)
    TN = np.sum(index_bothEmpty)
    FP = np.sum(preNonEmpty_TarNot)
    FN = np.sum(preEmpty_TarNot)
    
    acc = (TP + TN) / (TP + TN + FP + FN)
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    
    f = open("npyFolder/"+model+".txt", "w")
    f.write("model="+model+\
            
            "\n\nng_numOfNegativeVoxels="+str(ng_negative_vox)+\
            "\ngm_numOfNegativeVoxels="+str(gm_negative_vox)+\
            
            "\n\nng_overallMSE="+str(ng_mse)+\
            "\ngm_overallMSE="+str(gm_mse)+\
            "\nng_MSEforEitherNonEmptyVoxels="+str(ng_mse_eitherNonEmpty)+\
            "\ngm_MSEforEitherNonEmptyVoxels="+str(gm_mse_eitherNonEmpty)+\
            
            "\n\ntotalVox="+str(totalVox)+\
            "\nng_nonEmptyVoxInPrediction="+str(ng_nonEmptyVox_pre)+\
            "\nng_nonEmptyVoxInTarget="+str(ng_nonEmptyVox_tar)+\
            "\ngm_nonEmptyVoxInPrediction="+str(gm_nonEmptyVox_pre)+\
            "\ngm_nonEmptyVoxInTarget="+str(gm_nonEmptyVox_tar)+\
            
            "\n\ntake ng_pre as ground truth:"+\
            "\nTP="+str(TP)+\
            "\nTN="+str(TN)+\
            "\nFP="+str(FP)+\
            "\nFN="+str(FN)+\
            "\n\naccuracy="+str(acc)+\
            "\nrecall="+str(recall)+\
            "\nprecision="+str(precision))
    f.close()
    '''
    
    maxbin = int(np.amax([np.amax(cube_1), np.amax(cube_1_rounded), np.amax(cube_HOD), np.amax(cube_T)])) + 1

    bw_1 = 0.2
    bins_1 = np.arange(0, maxbin, 1)
    bins_1_rounded = np.arange(0, maxbin, 1)
    bins_HOD = np.arange(0, maxbin, 1)
    bins_T = np.arange(0, maxbin, 1)

    plt.figure(figsize=[9,9])
    plt.rc("font", size=14)
    
    plt.axvspan(1, 2, color='#FFF5A8', alpha=0.4)
    plt.hist(cube_1.flatten(), bins_1, alpha=1, label="CNN cascade w/o rounding", histtype='step',  density=True, color="#F338FC")
    plt.hist(cube_1_rounded.flatten(), bins_1_rounded, alpha=1, label="CNN cascade w/ rounding", histtype='step',  density=True, color="#ED0000")
    
    
    
    plt.hist(cube_HOD.flatten(), bins_HOD, alpha=1, label="HOD", histtype='step',  density=True, color="#0BB50B")
    plt.hist(cube_T.flatten(), bins_T, alpha=1, label="Target (TNG300-1)", histtype='step', density=True,color="#000ED4")

    plt.legend()
    plt.ylabel("PDF of Galaxy Number Density Field")
    plt.xlabel("Number of Galaxies in Voxel")
    plt.xscale('log')
    plt.yscale('log')
    #plt.title("Distribution of Galaxies in Voxels")
    plt.savefig(sim+"/npyFolder/ngDistHist.png")
    plt.close()

    '''
    bins = [10 ** float(n) for n in np.arange(-7.5, 4, 0.125)]
    plt.figure()
    plt.hist(gm_tar.flatten(), bins, alpha=0.5, label="Target")
    plt.hist(gm_pre.flatten(), bins, alpha=0.5, label="Prediction")
    plt.legend()
    plt.xlabel("Total Mass of Galaxies in the Voxel (10^10 Mo/h)")
    plt.ylabel("Number of Voxels")
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Distribution of Masses in Voxels")
    plt.savefig(sim+"/npyFolder/"+model_gm+"_gmDistHist.png")
    plt.close()
    '''

elif i == 14:
    cube_D = np.load("Data/"+sim+"/DarkMatter_Data_Masses/dm_"+sim+"-1-Dark_Masses_grid.npy")[448:, 448:, 448:]
    cube_T = np.load("Data/" + sim + "/StellarSubhalos_Data_StarsMassesNum/subhalos_" + sim + "-1_flagged_StarsMassesNum_grid.npy")[448:, 448:, 448:]
    cube_1 = np.load(sim + "/npyFolder/p=0_lw=500_s=13579_e=27_p=1_lw=1d05_s=12345_e=36_ng.npy")
    cube_1_rounded = np.load(sim + "/npyFolder/p=0_lw=500_s=13579_e=27_p=1_lw=2d2_s=12345_e=45_rounded_ng.npy")
    cube_HOD = np.load(sim + "/npyFolder/HODcube.npy")
    
    ind = np.where(cube_T==np.amax(cube_T))
    
    n = 25
    
    cube_D = cube_D[ind[0][0], ind[1][0]-n:ind[1][0]+n, ind[2][0]-n:ind[2][0]+n]
    cube_T = cube_T[ind[0][0], ind[1][0]-n:ind[1][0]+n, ind[2][0]-n:ind[2][0]+n]
    #cube_T[cube_T > 2] = 2
    cube_1 = cube_1[ind[0][0], ind[1][0]-n:ind[1][0]+n, ind[2][0]-n:ind[2][0]+n]
    #cube_1[cube_1 > 2] = 2
    cube_1_rounded = cube_1_rounded[ind[0][0], ind[1][0]-n:ind[1][0]+n, ind[2][0]-n:ind[2][0]+n]
    #cube_1_rounded[cube_1_rounded > 2] = 2
    cube_HOD = cube_HOD[ind[0][0], ind[1][0]-n:ind[1][0]+n, ind[2][0]-n:ind[2][0]+n]
    #cube_HOD[cube_HOD > 2] = 2
    
    grid = plt.GridSpec(1, 5)
    
    plt.figure(figsize=[25,5])
    plt.rc("font", size=14)
    
    #norm=Normalize(vmin=0., vmax=1., clip=False)
    #cmap = plt.get_cmap('gnuplot2')
    plt.subplot(grid[0, 0])
    plt.imshow(cube_D, norm=Normalize(vmin=0., vmax=0.5, clip=False), cmap = plt.get_cmap('gray'))
    plt.title("Input\n(Dark Matter/TNG300-1-Dark)")
    plt.xlabel("Voxel")
    plt.ylabel("Voxel")
    #plt.colorbar()
    plt.subplot(grid[0, 1])
    plt.imshow(cube_T, vmin=0, vmax=2)
    plt.title("Target\n(Galaxies/TNG300-1)")
    plt.axis("off")
    plt.subplot(grid[0, 2])
    plt.imshow(cube_1, vmin=0, vmax=2)
    plt.title("CNN cascade")
    plt.axis("off")
    plt.subplot(grid[0, 3])
    plt.imshow(cube_1_rounded, vmin=0, vmax=2)
    plt.title("CNN cascade\nw/ rounding")
    plt.axis("off")
    plt.subplot(grid[0, 4])
    plt.imshow(cube_HOD, vmin=0, vmax=2)
    plt.title("HOD")
    plt.axis("off")
    #plt.colorbar()
    
    plt.savefig(sim+"/npyFolder/heatmapS.png")
    
elif i == 15:
    cube_Da = np.load("Data/"+sim+"/DarkMatter_Data_Masses/dm_"+sim+"-1-Dark_Masses_grid.npy")[448:, 448:, 448:]
    cube_Ta = np.load("Data/" + sim + "/StellarSubhalos_Data_StarsMassesNum/subhalos_" + sim + "-1_flagged_StarsMassesNum_grid.npy")[448:, 448:, 448:]
    cube_1a = np.load(sim + "/npyFolder/p=0_lw=500_s=13579_e=27_p=1_lw=1d05_s=12345_e=36_ng.npy")
    cube_1_roundeda = np.load(sim + "/npyFolder/p=0_lw=500_s=13579_e=27_p=1_lw=2d2_s=12345_e=45_rounded_ng.npy")
    cube_HODa = np.load(sim + "/npyFolder/HODcube.npy")
    
    ind = np.where(cube_Ta==np.amax(cube_Ta))
    
    n = 75
    
    cube_D = cube_Da[ind[0][0], ind[1][0]-n:ind[1][0]+n, ind[2][0]-n:ind[2][0]+n]
    cube_T = cube_Ta[ind[0][0], ind[1][0]-n:ind[1][0]+n, ind[2][0]-n:ind[2][0]+n]
    #cube_T[cube_T > 2] = 2
    cube_1 = cube_1a[ind[0][0], ind[1][0]-n:ind[1][0]+n, ind[2][0]-n:ind[2][0]+n]
    #cube_1[cube_1 > 2] = 2
    cube_1_rounded = cube_1_roundeda[ind[0][0], ind[1][0]-n:ind[1][0]+n, ind[2][0]-n:ind[2][0]+n]
    #cube_1_rounded[cube_1_rounded > 2] = 2
    cube_HOD = cube_HODa[ind[0][0], ind[1][0]-n:ind[1][0]+n, ind[2][0]-n:ind[2][0]+n]
    #cube_HOD[cube_HOD > 2] = 2
    
    #grid = plt.GridSpec(2, 5)
    rect1 = patches.Rectangle((50,50),50,50,linewidth=0.5,edgecolor='#FFFFFF',facecolor='none')
    rect2 = patches.Rectangle((50,50),50,50,linewidth=0.5,edgecolor='#FFFFFF',facecolor='none')
    rect3 = patches.Rectangle((50,50),50,50,linewidth=0.5,edgecolor='#FFFFFF',facecolor='none')
    rect4 = patches.Rectangle((50,50),50,50,linewidth=0.5,edgecolor='#FFFFFF',facecolor='none')
    rect5 = patches.Rectangle((50,50),50,50,linewidth=0.5,edgecolor='#FFFFFF',facecolor='none')
    
    fig = plt.figure(figsize=[25,10])
    plt.rc("font", size=14)
    
    #norm=Normalize(vmin=0., vmax=1., clip=False)
    #cmap = plt.get_cmap('gnuplot2')
    ax0 = fig.add_subplot(251)
    ax0.imshow(cube_D, norm=Normalize(vmin=0., vmax=0.5, clip=False), cmap = plt.get_cmap('gray'))
    ax0.set_title("Input\n(Dark Matter/TNG300-1-Dark)")
    ax0.set_xlabel("Voxel")
    ax0.set_ylabel("Voxel")
    ax0.add_patch(rect1)
    #plt.colorbar()
    
    ax1 = fig.add_subplot(252)
    ax1.imshow(cube_T, vmin=0, vmax=2)
    ax1.set_title("Target\n(Galaxies/TNG300-1)")
    ax1.axis("off")
    ax1.add_patch(rect2)
    
    ax2 = fig.add_subplot(253)
    ax2.imshow(cube_1, vmin=0, vmax=2)
    ax2.set_title("CNN cascade\n w/o rounding")
    ax2.axis("off")
    ax2.add_patch(rect3)
    
    ax3 = fig.add_subplot(254)
    ax3.imshow(cube_1_rounded, vmin=0, vmax=2)
    ax3.set_title("CNN cascade\nw/ rounding")
    ax3.axis("off")
    ax3.add_patch(rect4)
    
    ax4 = fig.add_subplot(255)
    ax4.imshow(cube_HOD, vmin=0, vmax=2)
    ax4.set_title("HOD")
    ax4.axis("off")
    ax4.add_patch(rect5)
    #plt.colorbar()
    
    n = 25
    
    cube_D = cube_Da[ind[0][0], ind[1][0]-n:ind[1][0]+n, ind[2][0]-n:ind[2][0]+n]
    cube_T = cube_Ta[ind[0][0], ind[1][0]-n:ind[1][0]+n, ind[2][0]-n:ind[2][0]+n]
    #cube_T[cube_T > 2] = 2
    cube_1 = cube_1a[ind[0][0], ind[1][0]-n:ind[1][0]+n, ind[2][0]-n:ind[2][0]+n]
    #cube_1[cube_1 > 2] = 2
    cube_1_rounded = cube_1_roundeda[ind[0][0], ind[1][0]-n:ind[1][0]+n, ind[2][0]-n:ind[2][0]+n]
    #cube_1_rounded[cube_1_rounded > 2] = 2
    cube_HOD = cube_HODa[ind[0][0], ind[1][0]-n:ind[1][0]+n, ind[2][0]-n:ind[2][0]+n]
    #cube_HOD[cube_HOD > 2] = 2
     
    #norm=Normalize(vmin=0., vmax=1., clip=False)
    #cmap = plt.get_cmap('gnuplot2')
    ax5 = fig.add_subplot(256)
    ax5.imshow(cube_D, norm=Normalize(vmin=0., vmax=0.5, clip=False), cmap = plt.get_cmap('gray'))
    ax5.set_xlabel("Voxel")
    ax5.set_ylabel("Voxel")
    #plt.colorbar()
    ax6 = fig.add_subplot(257)
    ax6.imshow(cube_T, vmin=0, vmax=2)
    ax6.axis("off")
    
    ax7 = fig.add_subplot(258)
    ax7.imshow(cube_1, vmin=0, vmax=2)
    ax7.axis("off")
    
    ax8 = fig.add_subplot(259)
    ax8.imshow(cube_1_rounded, vmin=0, vmax=2)
    ax8.axis("off")
    
    ax9 = fig.add_subplot(2,5,10)
    ax9.imshow(cube_HOD, vmin=0, vmax=2)
    ax9.axis("off")
    #plt.colorbar()
    
    plt.savefig(sim+"/npyFolder/heatmapLS.png")

elif i == 16:
    cube_Da = np.load("Data/"+sim+"/DarkMatter_Data_Masses/dm_"+sim+"-1-Dark_Masses_grid.npy")[448:, 448:, 448:]
    cube_Ta = np.load("Data/" + sim + "/StellarSubhalos_Data_StarsMassesNum/subhalos_" + sim + "-1_flagged_StarsMassesNum_grid.npy")[448:, 448:, 448:]
    cube_1a = np.load(sim + "/npyFolder/p=0_lw=500_s=13579_e=27_p=1_lw=1d05_s=12345_e=36_ng.npy")
    cube_1_roundeda = np.load(sim + "/npyFolder/p=0_lw=500_s=13579_e=27_p=1_lw=2d2_s=12345_e=45_rounded_ng.npy")
    cube_HODa = np.load(sim + "/npyFolder/HODcube.npy")
    
    ind = np.where(cube_Ta==np.amax(cube_Ta))
    #print(np.amax(cube_Ta))
    print(ind)
    
    n = 75
    m = 180
    w = 100
    
    cube_D = cube_Da[ind[0][0], ind[1][0]-m-n:ind[1][0]-m+n, ind[2][0]-w-n:ind[2][0]-w+n]
    cube_T = cube_Ta[ind[0][0], ind[1][0]-m-n:ind[1][0]-m+n, ind[2][0]-w-n:ind[2][0]-w+n]
    #cube_T[cube_T > 2] = 2
    cube_1 = cube_1a[ind[0][0], ind[1][0]-m-n:ind[1][0]-m+n, ind[2][0]-w-n:ind[2][0]-w+n]
    #cube_1[cube_1 > 2] = 2
    cube_1_rounded = cube_1_roundeda[ind[0][0], ind[1][0]-m-n:ind[1][0]-m+n, ind[2][0]-w-n:ind[2][0]-w+n]
    #cube_1_rounded[cube_1_rounded > 2] = 2
    cube_HOD = cube_HODa[ind[0][0], ind[1][0]-m-n:ind[1][0]-m+n, ind[2][0]-w-n:ind[2][0]-w+n]
    #cube_HOD[cube_HOD > 2] = 2
    
    print(np.amax(cube_D))
    print(np.amax(cube_T))
    print(np.amax(cube_1))
    print(np.amax(cube_HOD))
    
    #grid = plt.GridSpec(2, 5)
    rect1 = patches.Rectangle((11,11),8,8,linewidth=0.7,edgecolor='#FFFFFF',facecolor='none')
    rect2 = patches.Rectangle((55,55),40,40,linewidth=0.7,edgecolor='#FFFFFF',facecolor='none')
    rect3 = patches.Rectangle((55,55),40,40,linewidth=0.7,edgecolor='#FFFFFF',facecolor='none')
    rect4 = patches.Rectangle((55,55),40,40,linewidth=0.7,edgecolor='#FFFFFF',facecolor='none')
    rect5 = patches.Rectangle((55,55),40,40,linewidth=0.7,edgecolor='#FFFFFF',facecolor='none')
    
    fig = plt.figure(figsize=[20,10])
    plt.rc("font", size=14)
    plt.rc('axes', titlesize=17.4)
    plt.rc('axes', labelsize=17.4)
    #plt.rc('legend', fontsize=17)
    #plt.rc('xtick', labelsize=13)    
    #plt.rc('ytick', labelsize=13)
    
    #norm=Normalize(vmin=0., vmax=1., clip=False)
    #cmap = plt.get_cmap('gnuplot2')
    ax0 = fig.add_subplot(241)
    ax0.imshow(cube_D, norm=Normalize(vmin=0., vmax=0.5, clip=False), cmap = plt.get_cmap('gray'), extent=[0,205/1024*150,0,205/1024*150], origin='lower')
    ax0.set_title("Input\n(Dark Matter/TNG300-1-Dark)")
    ax0.set_xlabel("Mpc/h")
    ax0.set_ylabel("Mpc/h")
    ax0.add_patch(rect1)
    #plt.colorbar()
    
    ax1 = fig.add_subplot(242)
    ax1.imshow(cube_T, vmin=0, vmax=2, origin='lower')
    ax1.set_title("Target\n(Galaxies/TNG300-1)")
    ax1.axis("off")
    ax1.add_patch(rect2)
    
    ax2 = fig.add_subplot(243)
    ax2.imshow(cube_1, vmin=0, vmax=2, origin='lower')
    ax2.set_title("Cascade Model")
    ax2.axis("off")
    ax2.add_patch(rect3)
    '''
    ax3 = fig.add_subplot(244)
    ax3.imshow(cube_1_rounded, vmin=0, vmax=2)
    ax3.set_title("CNN cascade\nw/ rounding")
    ax3.axis("off")
    #ax3.add_patch(rect4)
    '''
    ax4 = fig.add_subplot(244)
    ax4.imshow(cube_HOD, vmin=0, vmax=2, origin='lower')
    ax4.set_title("HOD")
    ax4.axis("off")
    ax4.add_patch(rect5)
    #plt.colorbar()
    
    n = 20
    
    cube_D = cube_Da[ind[0][0], ind[1][0]-m-n:ind[1][0]-m+n, ind[2][0]-w-n:ind[2][0]-w+n]
    cube_T = cube_Ta[ind[0][0], ind[1][0]-m-n:ind[1][0]-m+n, ind[2][0]-w-n:ind[2][0]-w+n]
    #cube_T[cube_T > 2] = 2
    cube_1 = cube_1a[ind[0][0], ind[1][0]-m-n:ind[1][0]-m+n, ind[2][0]-w-n:ind[2][0]-w+n]
    #cube_1[cube_1 > 2] = 2
    cube_1_rounded = cube_1_roundeda[ind[0][0], ind[1][0]-m-n:ind[1][0]-m+n, ind[2][0]-w-n:ind[2][0]-w+n]
    #cube_1_rounded[cube_1_rounded > 2] = 2
    cube_HOD = cube_HODa[ind[0][0], ind[1][0]-m-n:ind[1][0]-m+n, ind[2][0]-w-n:ind[2][0]-w+n]
    #cube_HOD[cube_HOD > 2] = 2
     
    #norm=Normalize(vmin=0., vmax=1., clip=False)
    #cmap = plt.get_cmap('gnuplot2')
    ax5 = fig.add_subplot(245)
    ax5.imshow(cube_D, norm=Normalize(vmin=0., vmax=0.5, clip=False), cmap = plt.get_cmap('gray'), extent=[0,205/1024*n*2,0,205/1024*n*2], origin='lower')
    ax5.set_xlabel("Mpc/h")
    ax5.set_ylabel("Mpc/h")
    #plt.colorbar()
    ax6 = fig.add_subplot(246)
    ax6.imshow(cube_T, vmin=0, vmax=2, origin='lower')
    ax6.axis("off")
    
    ax7 = fig.add_subplot(247)
    ax7.imshow(cube_1, vmin=0, vmax=2, origin='lower')
    ax7.axis("off")
    '''
    ax8 = fig.add_subplot(248)
    ax8.imshow(cube_1_rounded, vmin=0, vmax=2)
    ax8.axis("off")
    '''
    ax9 = fig.add_subplot(248)
    ax9.imshow(cube_HOD, vmin=0, vmax=2, origin='lower')
    ax9.axis("off")
    #plt.colorbar()
    
    plt.savefig(sim+"/npyFolder/heatmapLS_NR_PRESENT.png")
else:
    gm_tar = np.load("Data/"+sim+"/StellarSubhalos_Data_StarsMasses/subhalos_"+sim+"-1_StarsMasses_grid.npy")[512:, 512:, 512:]
    gm_pre = np.load(sim+"/npyFolder/"+model_gm+".npy")[512:, 512:, 512:]
    dm = np.load("Data/"+sim+"/DarkMatter_Data_Masses/dm_"+sim+"-1-Dark_Masses_grid.npy")[512:, 512:, 512:]
    
    dm1 = np.zeros((16, 16, 16))
    gm_tar1 = np.zeros((16, 16, 16))
    gm_pre1 = np.zeros((16, 16, 16))
    
    pos = list(np.arange(0, 512, 32))
    ranges = list(product(pos, repeat=3))
    for ID in ranges:
        dm1[int(ID[0]/32), int(ID[1]/32), int(ID[2]/32)] = dm[ID[0]:ID[0]+32, ID[1]:ID[1]+32, ID[2]:ID[2]+32].sum()
        gm_tar1[int(ID[0]/32), int(ID[1]/32), int(ID[2]/32)] = gm_tar[ID[0]:ID[0]+32, ID[1]:ID[1]+32, ID[2]:ID[2]+32].sum()
        gm_pre1[int(ID[0]/32), int(ID[1]/32), int(ID[2]/32)] = gm_pre[ID[0]:ID[0]+32, ID[1]:ID[1]+32, ID[2]:ID[2]+32].sum()
           
    dm = dm1
    gm_tar = gm_tar1
    gm_pre = gm_pre1
       
    '''
    ng_tar = np.load("Data/"+sim+"/StellarSubhalos_Data_StarsMassesNum/subhalos_"+sim+"-1_StarsMassesNum_grid.npy")[448:, 448:, 448:].flatten()
    ng_pre = np.load("npyFolder/"+model_ng+".npy")[448:, 448:, 448:].flatten()
    '''
    
    gm_pre_masked = gm_pre[gm_pre > 0]
    dm_gm_pre = dm[gm_pre > 0]
    
    gm_tar_masked = gm_tar[gm_tar > 0]
    dm_gm_tar = dm[gm_tar > 0]
    
    '''
    ng_pre_masked = ng_pre[ng_pre > 0]
    dm_ng_pre = dm[ng_pre > 0]
    
    ng_tar_masked = ng_tar[ng_tar > 0]
    dm_ng_tar = dm[ng_tar > 0]
    '''
    
    plt.figure()
    plt.scatter(dm_gm_tar, gm_tar_masked/dm_gm_tar, s=1, label="Target")
    plt.scatter(dm_gm_pre, gm_pre_masked/dm_gm_pre, s=1, label="Prediction")
    plt.xlabel("Dark Matter Mass in Voxel (10^10 Mo/h)")
    plt.ylabel("Total Mass of Galaxies / Dark Matter Mass in Voxel (10^10 Mo/h)")
    plt.legend()
    #plt.title("Mass of Galaxies vs Mass of Dark Matter")
    plt.savefig(sim+"/npyFolder/"+model_gm+"_gmOdmVSdm")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(left=100)
    plt.ylim(bottom=0.0000001)
    plt.savefig(sim+"/npyFolder/"+model_gm+"_gmOdmVSdm_log")
    plt.close()
    
    plt.figure()
    plt.scatter(dm_gm_tar, gm_tar_masked, s=1, label="Target")
    plt.scatter(dm_gm_pre, gm_pre_masked, s=1, label="Prediction")
    plt.xlabel("Dark Matter Mass in Voxel (10^10 Mo/h)")
    plt.ylabel("Total Mass of Galaxies in Voxel (10^10 Mo/h)")
    plt.legend()
    #plt.title("Mass of Galaxies vs Mass of Dark Matter")
    plt.savefig(sim+"/npyFolder/"+model_gm+"_gmVSdm")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(left=100)
    plt.ylim(bottom=0.00001)
    plt.savefig(sim+"/npyFolder/"+model_gm+"_gmVSdm_log")
    plt.close()
    
    '''
    plt.figure()
    
    plt.scatter(dm_ng_tar, ng_tar_masked, s=1, label="Target")
    plt.scatter(dm_ng_pre, ng_pre_masked, s=1, label="Prediction")
    plt.xlabel("Dark Matter Mass in Voxel (10^10 Mo/h)")
    plt.ylabel("Total Number of Galaxies in Voxel (10^10 Mo/h)")
    plt.legend()
    plt.title("Number of Galaxies vs Number of Dark Matter")
    plt.savefig("npyFolder/"+model+"_ngVSdm_r")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(left=0.1)
    plt.ylim(bottom=0.1)
    plt.savefig("npyFolder/"+model+"_ngVSdm_log_r")
    '''