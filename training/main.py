import torch
import torch.backends.cudnn as cudnn
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F

import socket
import os
import numpy as np
from itertools import product
import time

from Dataset import Dataset
from Models import *
from train_f import *

seed = 24680  # random seed
sim = "TNG300"
epochs, lr, batch_size, num_workers = 50, 0.001, 16, 20
conv1_out, conv3_out, conv5_out = 6, 8, 10
weight_decay, data_size, print_freq = 0, "full", 400

phase = 1

if phase == 0:  # inception for non-empty voxels
    loss_weight, round, C_model = 500, None, None
    G_model = None
    save_name = "p=0_lw=%s_s=%s" % (str(loss_weight).replace(".", "d"), str(seed))
elif phase == 1:  # use phase 0's result to mask for a r2unet for number of galaxies;
    loss_weight, round, C_model = 0.95, None, "p=0_lw=500_s=13579_e=27"
    G_model = None
    save_name = C_model + "_" + "p=1_lw=%s_s=%s" % (str(loss_weight).replace(".", "d"), str(seed))
elif phase == 2:  # use phase 1's result to mask and as second channel for another r2unet for galaxy masses
    loss_weight, round, C_model = None, None, "p=0_lw=_s=_e="
    G_model = C_model + "_" + "p=1_lw=_s=_e="
    save_name = G_model + "_" + "p=2_lw=%s_r=%s_s=%s" % (str(loss_weight).replace(".", "d"), str(round), str(seed))

plot_label, txt_name = save_name, save_name

torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True

#cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_LOSS, VAL_LOSS = [], []
BEST_VAL_LOSS, BEST_RECALL, BEST_PRECISION, BEST_ACC = 1e20, 0, 0, 0
EPSILON = 1e-8

if len(save_name) > 0 and not os.path.exists(sim):
    os.makedirs(sim)


def initial_loss(train_loader, val_loader, model, criterion, phase):
    start_time = time.time()
    f = open(sim+"/"+txt_name+".txt", "a")
    
    train_losses, val_losses = AverageMeter(), AverageMeter()
    correct, total = 0, 0
    TPRs, FPRs = AverageMeter(), AverageMeter()

    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(train_loader):
            input = input.to(device).float()
            if phase == 0:
                target = target.to(device).long()
            else:
                target = target.to(device).float()
            output = model(input)

            loss = criterion(output, target)
            train_losses.update(loss.item(), input.size(0))

        for i, (input, target) in enumerate(val_loader):
            input = input.to(device).float()
            if phase == 0:
                target = target.to(device).long()
            else:
                target = target.to(device).float()
            output = model(input)

            if phase == 0:
                outputs = F.softmax(output, dim=1)
                predicted = outputs.max(1)[1]
                total += np.prod(target.shape)
                correct += predicted.eq(target.view_as(predicted)).sum().item()
                TPR, gp, FPR, gf = confusion_matrix_calc(predicted, target)
                TPRs.update(TPR, gp)
                FPRs.update(FPR, gf)

            loss = criterion(output, target)
            val_losses.update(loss.item(), input.size(0))

    if phase == 0:
        acc = correct / total
        recall = TPRs.avg
        precision = TPRs.sum / (TPRs.sum + FPRs.sum + EPSILON)

    TRAIN_LOSS.append(train_losses.avg)
    VAL_LOSS.append(val_losses.avg)

    initial_loss_time = (time.time() - start_time) / 60

    if phase == 0:
        f.write("\nInitial | Train Loss: {train_losses.avg:.8f} | Val Loss: {val_losses.avg:.8f} | Val Accuracy: {acc:.8f} | Val Recall: {recall:.8f} | Val Precision: {precision:.8f} | Time Used: {initial_loss_time:.3f}min\n".format(train_losses=train_losses, val_losses=val_losses, acc=acc, recall=recall, precision=precision, initial_loss_time=initial_loss_time))
    else:
        f.write("\nInitial | Train Loss: {train_losses.avg:.8f} | Val Loss: {val_losses.avg:.8f} | Time Used: {initial_loss_time:.3f}min\n".format(train_losses=train_losses, val_losses=val_losses, initial_loss_time=initial_loss_time))
    f.close()


def train(train_loader, model, criterion, epoch, phase, optimizer, print_freq):
    start_time = time.time()
    f = open(sim+"/"+txt_name+".txt", "a")

    losses = AverageMeter()

    model.train()
    for i, (input, target) in enumerate(train_loader):
        input = input.to(device).float()
        if phase == 0:
            target = target.to(device).long()
        else:
            target = target.to(device).float()
        output = model(input)

        loss = criterion(output, target)
        losses.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % print_freq == 0:
                f.write("\nEpoch {0} | [Batch {1} / {2}] | Train Loss: {losses.avg:.8f}".format(epoch, i, len(train_loader), losses=losses))

    train_time = (time.time() - start_time) / 60
    TRAIN_LOSS.append(losses.avg)
    f.write("\nEpoch {0} | Train Loss: {losses.avg:.8f} | Time Used: {train_time:.3f}min\n".format(epoch, losses=losses, train_time=train_time))
    f.close()


def validate(val_loader, model, criterion, epoch, phase, save_name):
    start_time = time.time()
    f = open(sim+"/"+txt_name+".txt", "a")

    global BEST_VAL_LOSS
    val_losses, TPRs, FPRs = AverageMeter(), AverageMeter(), AverageMeter()
    total, correct = 0, 0
    
    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input = input.to(device).float()
            if phase == 0:
                target = target.to(device).long()
            else:
                target = target.to(device).float()
            output = model(input)

            if phase == 0:
                outputs = F.softmax(output, dim=1)
                predicted = outputs.max(1)[1]
                total += np.prod(target.shape)
                correct += predicted.eq(target.view_as(predicted)).sum().item()
                TPR, gp, FPR, gf = confusion_matrix_calc(predicted, target)
                TPRs.update(TPR, gp)
                FPRs.update(FPR, gf)

            loss = criterion(output, target)
            val_losses.update(loss.item(), input.size(0))

    if phase == 0:
        acc = correct / total
        recall = TPRs.avg
        precision = TPRs.sum / (TPRs.sum + FPRs.sum + EPSILON)

    if val_losses.avg < BEST_VAL_LOSS:
        BEST_VAL_LOSS = val_losses.avg
        if len(save_name) > 0:
            torch.save(model.state_dict(), sim+"/"+str(save_name)+"_e="+str(epoch)+".pth")
    
    VAL_LOSS.append(val_losses.avg)

    val_time = (time.time() - start_time) / 60
    if phase == 0:
        f.write("\nValidation | Val. Loss: {val_losses.avg:.8f} | Val. Accuracy: {acc:.8f} | Val. Recall: {recall:.8f} | Val. Precision: {precision:.8f} | Time Used: {val_time:.3f}min\n".format(val_losses=val_losses, acc=acc, recall=recall, precision=precision, val_time=val_time))
    else:
        f.write("\nValidation | Val. Loss: {val_losses.avg:.8f} | Time Used: {val_time:.3f}min\n".format(val_losses=val_losses, val_time=val_time))
    f.close()

def main():
    start_time = time.time()
    f = open(sim+"/"+txt_name+".txt", "w")

    f.write("hostname=" + str(socket.gethostname()) +
            "\ndevice=" + str(device) +

            "\n\nseed=" + str(seed) +
            "\nsim=" + sim +

            "\n\nepochs=" + str(epochs) +
            "\nlr=" + str(lr) +
            "\nbatch_size=" + str(batch_size) +
            "\nnum_workers=" + str(num_workers) +

            "\n\nconv1_out=" + str(conv1_out) +
            "\nconv3_out=" + str(conv3_out) +
            "\nconv5_out=" + str(conv5_out) +

            "\n\nweight_decay=" + str(weight_decay) +
            "\ndata_size=" + data_size +
            "\nprint_freq=" + str(print_freq) +

            "\n\nphase=" + str(phase) +

            "\n\nloss_weight=" + str(loss_weight) +
            "\nround=" + str(round) +
            "\nC_model=" + str(C_model) +
            "\nG_model=" + str(G_model) +
            "\nsave_name=" + save_name +

            "\n\nplot_label=" + plot_label +

            "\n")
    f.close()

    if data_size == "full":
        data_range = 1024
    pos = list(np.arange(0, data_range, 32))
    ranges = list(product(pos, repeat=3))
    train_data, val_data, test_data = [], [], []
    for i in ranges:
        if i[0] <= 416 and i[1] <= 416:  # 19.1%
            val_data.append(i)
        elif i[0] >= 448 and i[1] >= 448 and i[2] >= 448:  # 18*18*18; 17.8%
            test_data.append(i)
        else:
            train_data.append(i)

    if phase == 2:
        dm_box = np.load("Data/"+sim+"/DarkMatter_Data_Masses/dm_"+sim+"-1-Dark_Masses_grid.npy")
        ng_box = None
        gm_box = np.load("Data/"+sim+"/StellarSubhalos_Data_StarsMasses/subhalos_"+sim+"-1_flagged_StarsMasses_grid.npy")
    else:
        dm_box = np.load("Data/"+sim+"/DarkMatter_Data_Masses/dm_"+sim+"-1-Dark_Masses_grid.npy")
        ng_box = np.load("Data/"+sim+"/StellarSubhalos_Data_StarsMassesNum/subhalos_"+sim+"-1_flagged_StarsMassesNum_grid.npy")
        gm_box = None

    params = {"batch_size": batch_size, "shuffle": True, "num_workers": num_workers}
    training_set, validation_set = Dataset(train_data, phase=phase, flipBox=True, dm_box=dm_box, ng_box=ng_box, gm_box=gm_box), Dataset(val_data, phase=phase, flipBox=True, dm_box=dm_box, ng_box=ng_box, gm_box=gm_box)
    training_generator, validation_generator = data.DataLoader(training_set, **params), data.DataLoader(validation_set, **params)

    if phase == 0:
        model = Inception(1, conv1_out, conv3_out, conv5_out, pool_out=3).to(device)
    elif phase == 1:
        mask_model = Inception(1, conv1_out, conv3_out, conv5_out, pool_out=3).to(device)
        state_dict = torch.load(sim+"/"+C_model+".pth")
        mask_model.load_state_dict(state_dict)
        pred_model = R2Unet(1, 1, t=3, phase=1).to(device)
        
        model = masked_conv_1(mask_model, pred_model, thres=0.5).to(device)
    elif phase == 2:
        mask_model_0 = Inception(1, conv1_out, conv3_out, conv5_out, pool_out=3).to(device)
        state_dict_0 = torch.load(sim+"/"+C_model+".pth")
        mask_model_0.load_state_dict(state_dict_0)
        pred_model_1 = R2Unet(1, 1, t=3, phase=1).to(device)
        
        mask_model_1 = masked_conv_1(mask_model_0, pred_model_1, thres=0.5).to(device)
        
        state_dict_1 = torch.load(sim+"/"+G_model+".pth")
        mask_model_1.load_state_dict(state_dict_1)
        pred_model_2 = R2Unet(2, 1, t=3, phase=2).to(device)
        
        model = masked_conv_2(mask_model_1, pred_model_2, round=round).to(device)

    if phase == 0:
        criterion = nn.CrossEntropyLoss(weight=get_loss_weight(loss_weight, num_class=2)).to(device)
    #elif phase == 1:
    else:
        criterion = weighted_nn_loss(loss_weight)
    #elif phase == 2:
        #criterion = weighted_l1_loss(loss_weight)
        

    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay)

    initial_loss(training_generator, validation_generator, model, criterion, phase)
    for epoch in range(epochs):
        adjust_learning_rate(lr, optimizer, epoch)
        train(training_generator, model, criterion, epoch, phase, optimizer, print_freq)
        validate(validation_generator, model, criterion, epoch, phase, save_name)
        
        if len(plot_label) > 0:
          train_plot(TRAIN_LOSS, VAL_LOSS, plot_label, sim)
        
    overall_time = (time.time() - start_time) / 60
    f = open(sim+"/"+txt_name+".txt", "a")
    f.write("\nOverall Time Used: {overall_time:.3f}min".format(overall_time=overall_time))
    f.close()


if __name__ == "__main__":
    main()
