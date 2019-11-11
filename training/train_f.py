import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


def weighted_nn_loss(weight_ratio):
    def weighted(X, Y):  # X is output, Y is target
        base_loss = F.mse_loss(X, Y, reduction="sum")
        index = Y > 0
        plus_loss = F.mse_loss(X[index], Y[index], reduction="sum") if index.any() > 0 else 0
        total_loss = base_loss + (weight_ratio - 1) * plus_loss
        return total_loss / X.shape[0]
    return weighted


'''
def weighted_l1_loss(weight_ratio):
    def weighted(X, Y):  # X is output, Y is target
        base_loss = F.l1_loss(X, Y, reduction="sum")
        index = Y > 0
        plus_loss = F.l1_loss(X[index], Y[index], reduction="sum") if index.any() > 0 else 0
        total_loss = base_loss + (weight_ratio - 1) * plus_loss
        return total_loss / X.shape[0]
    return weighted
'''


def get_loss_weight(loss_weight, num_class):
    piece = 1 / ((num_class - 1) * loss_weight + 1)
    a = [1]
    a.extend([loss_weight] * (num_class - 1))
    return (torch.from_numpy(piece * np.array(a))).float()


def adjust_learning_rate(lr0, optimizer, epoch):
    lr = lr0 * (0.5 ** (epoch // 4))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def train_plot(train_loss, val_loss, plot_label, sim):
    plt.figure()
    plt.plot(train_loss, label="Training Loss", linewidth="1")
    plt.plot(val_loss, label="Validation Loss", linewidth="1")
    plt.legend()
    plt.savefig(sim+"/"+plot_label+".png")
    plt.close()


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def confusion_matrix_calc(pred, Y):
    Y_index_nonEm = Y > 0

    TP = torch.sum(pred[Y_index_nonEm] > 0).item()
    FP = torch.sum(pred[~Y_index_nonEm] > 0).item()
    ground_P = Y[Y_index_nonEm].numel()
    ground_F = Y[~Y_index_nonEm].numel()

    return TP/ground_P, ground_P, FP/ground_F, ground_F
