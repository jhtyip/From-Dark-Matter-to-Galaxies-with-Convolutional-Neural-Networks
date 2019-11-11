import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv3d(nn.Module):
    def __init__(self, in_ch, out_ch, **kwargs):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm3d(out_ch, track_running_stats=True)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.batchnorm(x1)

        return F.relu(x2, inplace=True)


class InceptionE(nn.Module):  # inception module
    def __init__(self, in_ch, conv1_out, conv3_out, conv5_out, pool_out=3):
        super(InceptionE, self).__init__()
        self.branch1x1 = BasicConv3d(in_ch, conv1_out, kernel_size=1)
        self.branch3x3 = BasicConv3d(in_ch, conv3_out, kernel_size=3, padding=1)
        self.branch5x5 = BasicConv3d(in_ch, conv5_out, kernel_size=5, padding=2)
        self.branch_pool = BasicConv3d(in_ch, pool_out, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(F.avg_pool3d(x, kernel_size=3, stride=1, padding=1))

        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]

        return torch.cat(outputs, 1)


class Inception(nn.Module):
    def __init__(self, in_ch, conv1_out, conv3_out, conv5_out, pool_out=3):
        super(Inception, self).__init__()
        self.incep = InceptionE(in_ch, conv1_out=conv1_out, conv3_out=conv3_out, conv5_out=conv5_out, pool_out=pool_out)
        conv_in = conv1_out + conv3_out + conv5_out + pool_out
        self.conv1 = BasicConv3d(conv_in, conv_in, kernel_size=3, padding=1)
        self.conv2 = BasicConv3d(conv_in, conv_in//2, kernel_size=3, padding=1)
        self.conv3 = BasicConv3d(conv_in//2, 2, kernel_size=1)
        
    def forward(self, x):
        x1 = self.incep(x)
        x2 = self.conv1(x1)
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)

        return x4


class Recurrent_Conv(nn.Module):
    def __init__(self, out_ch, t):  # t is the number of times of re-conv
        super(Recurrent_Conv, self).__init__()
        self.t = t
        self.conv = nn.Sequential(nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU(inplace=True))

    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)
            x1 = self.conv(x + x1)

        return x1


class R2CNN(nn.Module):
    def __init__(self, in_ch, out_ch, t):
        super(R2CNN, self).__init__()
        self.initial_conv = nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.RCNN = nn.Sequential(Recurrent_Conv(out_ch, t), Recurrent_Conv(out_ch, t))  # 2 layers of Recurrent_Conv

    def forward(self, x):
        x1 = self.initial_conv(x)
        x2 = self.RCNN(x1)

        return x1 + x2


class R2Unet(nn.Module):
    def __init__(self, in_ch, out_ch, t=3, phase=1):
        super(R2Unet, self).__init__()
        self.phase = phase

        self.r2cnn1 = R2CNN(in_ch, 32, t)
        self.maxPool1 = nn.MaxPool3d(2)
        self.r2cnn2 = R2CNN(32, 64, t)
        self.maxPool2 = nn.MaxPool3d(2)
        self.r2cnn3 = R2CNN(64, 128, t)
        self.maxPool3 = nn.MaxPool3d(2)
        self.r2cnn4 = R2CNN(128, 256, t)
        self.up_conv1 = self.up_conv_layer(256, 128, 3, 2, 1, 1)
        self.r2cnn5 = R2CNN(256, 64, t)
        self.up_conv2 = self.up_conv_layer(64, 64, 3, 2, 1, 1)
        self.r2cnn6 = R2CNN(128, 32, t)
        self.up_conv3 = self.up_conv_layer(32, 32, 3, 2, 1, 1)
        self.r2cnn7 = R2CNN(32, 16, t)
        self.conv11 = nn.Conv3d(16, out_ch, kernel_size=1, stride=1, padding=0)

    def up_conv_layer(self, in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True):
        layers = nn.Sequential(nn.ConvTranspose3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias), nn.ReLU())
        return layers

    def forward(self, x):
        x1 = self.r2cnn1(x)
        x2 = self.maxPool1(x1)
        x3 = self.r2cnn2(x2)
        x4 = self.maxPool2(x3)
        x5 = self.r2cnn3(x4)
        x6 = self.maxPool3(x5)
        x7 = self.r2cnn4(x6)

        x8 = self.up_conv1(x7)
        x8 = torch.cat((x5, x8), dim=1)
        x9 = self.r2cnn5(x8)
        x10 = self.up_conv2(x9)
        x10 = torch.cat((x3, x10), dim=1)
        x11 = self.r2cnn6(x10)
        x12 = self.up_conv3(x11)
        x13 = self.r2cnn7(x12)
        x14 = self.conv11(x13)

        if self.phase == 0:
            return F.relu(x14, inplace=True)
        else:
            return F.relu(x14.squeeze(1), inplace=True)  # kill the 1 channel


class masked_conv_1(nn.Module):
    def __init__(self, first_pmodel, second_pmodel, thres=0.5):
        super(masked_conv_1, self).__init__()
        self.fp = first_pmodel
        for param in self.fp.parameters():
            param.requires_grad = False
        self.sp = second_pmodel
        self.thres = thres

    def forward(self, x):
        nonEmProb = F.softmax(self.fp(x), dim=1)[:, 1, :, :, :]  # take the [1] slice for non-empty probabilities
        mask_value = (nonEmProb > self.thres).float()
        output = mask_value * self.sp(x)

        return output


class masked_conv_2(nn.Module):
    def __init__(self, first_pmodel, second_pmodel, round=True):
        super(masked_conv_2, self).__init__()
        self.fp = first_pmodel
        for param in self.fp.parameters():
            param.requires_grad = False
        self.sp = second_pmodel
        self.round = round

    def forward(self, x):  # x=[N, 1, 32, 32, 32] of dark matter
        if self.round == True:
            numGal = self.fp(x).round().unsqueeze(1)  # numGal=[N, 1, 32, 32, 32]
        else:
            numGal = self.fp(x).unsqueeze(1)

        mask_value = (numGal.squeeze(1) > 0).float()
        x2 = torch.cat((x, numGal), dim=1)
        output = mask_value * self.sp(x2)

        return output
