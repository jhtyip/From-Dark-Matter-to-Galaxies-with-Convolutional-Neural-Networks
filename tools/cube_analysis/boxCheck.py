import torch
from torch.utils import data

import numpy as np
from itertools import product

from Dataset import Dataset
from Models import *
from train_f import *

import matplotlib.pyplot as plt

sim = "TNG300"
phase = 1

lw0, s0, e0 = 500, 13579, 27
lw1, s1, e1 = 2.3, 12345, 28
lw2, r2, s2, e2 = None, None, None, None

# whether to round the output
if phase == 1:
    round = True
elif phase == 2:
    round = False  # always False

n = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if phase == 1:
    mask_model = Inception(1, 6, 8, 10).to(device)
    state_dict_0 = torch.load(sim+"/"+"p=0_lw=%s_s=%s_e=%s.pth" % (str(lw0).replace(".", "d"), str(s0), str(e0)))
    mask_model.load_state_dict(state_dict_0)
    pred_model = R2Unet(1, 1, t=3, phase=1).to(device)
    model = masked_conv_1(mask_model, pred_model).to(device)
    state_dict_1 = torch.load(sim+"/"+"p=0_lw=%s_s=%s_e=%s_p=1_lw=%s_s=%s_e=%s.pth" % (str(lw0).replace(".", "d"), str(s0), str(e0), str(lw1).replace(".", "d"), str(s1), str(e1)))
    model.load_state_dict(state_dict_1)

    model_title = "p=0_lw=%s_s=%s_e=%s_p=1_lw=%s_s=%s_e=%s" % (str(lw0).replace(".", "d"), str(s0), str(e0), str(lw1).replace(".", "d"), str(s1), str(e1))
elif phase == 2:
    mask_model = Inception(1, 6, 8, 10).to(device)
    state_dict_0 = torch.load(sim+"/"+"p=0_lw=%s_s=%s_e=%s.pth" % (str(lw0).replace(".", "d"), str(s0), str(e0)))
    mask_model.load_state_dict(state_dict_0)
    pred_model = R2Unet(1, 1, t=3, phase=1).to(device)
    mask_model_1 = masked_conv_1(mask_model, pred_model).to(device)
    state_dict_1 = torch.load(sim+"/"+"p=0_lw=%s_s=%s_e=%s_p=1_lw=%s_s=%s_e=%s.pth" % (str(lw0).replace(".", "d"), str(s0), str(e0), str(lw1).replace(".", "d"), str(s1), str(e1)))
    mask_model_1.load_state_dict(state_dict_1)
    pred_model_2 = R2Unet(2, 1, t=3, phase=2).to(device)
    model = masked_conv_2(mask_model_1, pred_model_2, round=r2).to(device)
    state_dict_2 = torch.load(sim+"/"+"p=0_lw=%s_s=%s_e=%s_p=1_lw=%s_s=%s_e=%s_p=2_lw=%s_r=%s_s=%s_e=%s.pth" % (str(lw0).replace(".", "d"), str(s0), str(e0), str(lw1).replace(".", "d"), str(s1), str(e1), str(lw2).replace(".", "d"), str(r2), str(s2), str(e2)))
    model.load_state_dict(state_dict_2)

    model_title = "p=0_lw=%s_s=%s_e=%s_p=1_lw=%s_s=%s_e=%s_p=2_lw=%s_r=%s_s=%s_e=%s" % (str(lw0).replace(".", "d"), str(s0), str(e0), str(lw1).replace(".", "d"), str(s1), str(e1), str(lw2).replace(".", "d"), str(r2), str(s2), str(e2))

dm_box = np.load("Data/"+sim+"/DarkMatter_Data_Masses/dm_"+sim+"-1-Dark_Masses_grid.npy")
if phase == 1:
    ng_box = np.load("Data/"+sim+"/StellarSubhalos_Data_StarsMassesNum/subhalos_"+sim+"-1_flagged_StarsMassesNum_grid.npy")
    gm_box = None
elif phase == 2:
    ng_box = None
    gm_box = np.load("Data/"+sim+"/StellarSubhalos_Data_StarsMasses/subhalos_"+sim+"-1_flagged_StarsMasses_grid.npy")

pos = list(np.arange(0, 1024, 32))
ranges = list(product(pos, repeat=3))
train_data, val_data, test_data = [], [], []
for i in ranges:
    if i[0] <= 416 and i[1] <= 416:  # 14*14*32
        val_data.append(i)
    elif i[0] >= 448 and i[1] >= 448 and i[2] >= 448:  # 18*18*18
        test_data.append(i)
    else:
        train_data.append(i)

data_set = Dataset(val_data, phase=phase, flipBox=True, dm_box=dm_box, ng_box=ng_box, gm_box=gm_box)  # use validation set
params = {"batch_size": 16, "shuffle": True, "num_workers": 20}
generator = data.DataLoader(data_set, **params)

pred_ng = []
tar_ng = []

model.eval()
with torch.no_grad():
    inbox_ranges = list(product(list(np.arange(0, 32, n)), repeat=3))
    for i, (input, target) in enumerate(generator):
        input = input.to(device).float()
        target = target.to(device).float()
        if round == True:
            output = model(input).round()
        elif round == False:
            output = model(input)

        for j in inbox_ranges:
            pred_ng.extend(output[:, j[0]:j[0]+n, j[1]:j[1]+n, j[2]:j[2]+n].sum(1).sum(1).sum(1).tolist())
            tar_ng.extend(target[:, j[0]:j[0]+n, j[1]:j[1]+n, j[2]:j[2]+n].sum(1).sum(1).sum(1).tolist())

maxNum = max([np.amax(pred_ng), np.amax(tar_ng)])
maxNum10 = maxNum*0.1

MSE = np.sum((np.array(pred_ng) - np.array(tar_ng))**2) / len(pred_ng)

plt.figure()
plt.scatter(tar_ng, pred_ng, s=3)
plt.plot([-maxNum10, maxNum+maxNum10], [-maxNum10, maxNum+maxNum10], color="r", linewidth="1")
if phase == 1:
    if round == True:
        plt.title(sim+"_"+model_title+"\nTotal Rounded Number of Galaxies in each $\mathregular{"+str(n)+"^3}$ Subbox (Validation Set)\nNumber of Points: "+str(len(pred_ng))+"\nMSE: "+str(MSE), fontsize=8)
    elif round == False:
        plt.title(sim+"_"+model_title+"\nTotal Number of Galaxies in each $\mathregular{"+str(n)+"^3}$ Subbox (Validation Set)\nNumber of Points: "+str(len(pred_ng))+"\nMSE: "+str(MSE), fontsize=8)
    content = "ng"
elif phase == 2:
    plt.title(sim+"_"+model_title+"\nTotal Mass of Galaxies in each $\mathregular{"+str(n)+"^3}$ Subbox (Validation Set)\nNumber of Points: "+str(len(pred_ng))+"\nMSE: "+str(MSE), fontsize=8)
    content = "gm"
plt.xlabel("Target")
plt.ylabel("Prediction")
if round == True:
    plt.savefig(sim+"/"+model_title+"_rounded_"+str(content)+str(n)+"fullCheck")
elif round == False:
    plt.savefig(sim+"/"+model_title+"_"+str(content)+str(n)+"fullCheck")
plt.close()

plt.figure()
plt.scatter(tar_ng, pred_ng, s=3)
plt.plot([0.1, maxNum+maxNum10], [0.1, maxNum+maxNum10], color="r", linewidth="1")
if phase == 1:
    if round == True:
        plt.title(sim+"_"+model_title+"\nTotal Rounded Number of Galaxies in each $\mathregular{"+str(n)+"^3}$ Subbox (Validation Set)\nNumber of Points: "+str(len(pred_ng))+"\nMSE: "+str(MSE), fontsize=8)
    elif round == False:
        plt.title(sim+"_"+model_title+"\nTotal Number of Galaxies in each $\mathregular{"+str(n)+"^3}$ Subbox (Validation Set)\nNumber of Points: "+str(len(pred_ng))+"\nMSE: "+str(MSE), fontsize=8)
    content = "ng"
elif phase == 2:
    plt.title(sim+"_"+model_title+"\nTotal Mass of Galaxies in each $\mathregular{"+str(n)+"^3}$ Subbox (Validation Set)\nNumber of Points: "+str(len(pred_ng))+"\nMSE: "+str(MSE), fontsize=8)
plt.xscale("log") 
plt.yscale("log")
plt.xlabel("Target")
plt.ylabel("Prediction")
if round == True:
    plt.savefig(sim+"/"+model_title+"_rounded_"+str(content)+str(n)+"fullCheck_log.png")
elif round == False:
    plt.savefig(sim+"/"+model_title+"_"+str(content)+str(n)+"fullCheck_log.png")
plt.close()
