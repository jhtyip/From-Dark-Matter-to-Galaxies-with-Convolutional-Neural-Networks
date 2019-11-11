import torch
import numpy as np
import os
from itertools import product
from Dataset import Dataset
from Models import *

sim = "TNG300"
phase = 1

if not os.path.exists(sim+"/npyFolder"):
    os.makedirs(sim+"/npyFolder")

lw0, s0, e0 = 500, 13579, 27
lw1, s1, e1 = 1.05, 12345, 36
lw2, r2, s2, e2 = 2.35, True, 22, 24

# whether to round the output
if phase == 1:
    round = False
elif phase == 2:
    round = False  # always False

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
    content = "ng"
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
    content = "gm"

dm_box = np.load("Data/"+sim+"/DarkMatter_Data_Masses/dm_"+sim+"-1-Dark_Masses_grid.npy")

prediction = np.zeros((1024, 1024, 1024))
pos=list(np.arange(0, 1024, 32))
ranges = list(product(pos, repeat=3))

model.eval()
with torch.no_grad():
    for ID in ranges:
        input = torch.from_numpy(dm_box[ID[0]:ID[0]+32, ID[1]:ID[1]+32, ID[2]:ID[2]+32]).unsqueeze(dim=0).unsqueeze(dim=1).to(device).float()
        if phase == 1:
            if round == True:
                prediction[ID[0]:ID[0]+32, ID[1]:ID[1]+32, ID[2]:ID[2]+32] = model(input).cpu().numpy()[0].round()
            elif round == False:
                prediction[ID[0]:ID[0]+32, ID[1]:ID[1]+32, ID[2]:ID[2]+32] = model(input).cpu().numpy()[0]
        elif phase == 2:
            #prediction[ID[0]:ID[0] + 32, ID[1]:ID[1] + 32, ID[2]:ID[2] + 32] = model(input).cpu().numpy()[0]
            prediction[ID[0]:ID[0] + 32, ID[1]:ID[1] + 32, ID[2]:ID[2] + 32] = (model(input).cpu().numpy()[0] / 1000)**4

if round == True:
    np.save(sim+"/npyFolder/"+model_title+"_rounded_"+str(content)+".npy", prediction[448:, 448:, 448:])
elif round == False:
    np.save(sim+"/npyFolder/"+model_title+"_"+str(content)+".npy", prediction[448:, 448:, 448:])
