import torch
from torch.utils import data
import numpy as np


class Dataset(data.Dataset):
    def __init__(self, boxIDs, phase, flipBox=False, dm_box=None, ng_box=None, gm_box=None):
        self.boxIDs = boxIDs
        self.phase = phase
        self.flipBox = flipBox

        self.dm_box = dm_box
        self.ng_box = ng_box
        self.gm_box = gm_box

    def __len__(self):
        return len(self.boxIDs)

    def convert_to_class(self, num):
        if num > 0:
            return 1
        else:
            return 0

    def __getitem__(self, index):
        ID = self.boxIDs[index]

        inp_box = self.dm_box[ID[0]:ID[0]+32, ID[1]:ID[1]+32, ID[2]:ID[2]+32]

        if self.phase == 0:
            convert = np.vectorize(self.convert_to_class)
            tar_box = self.ng_box[ID[0]:ID[0]+32, ID[1]:ID[1]+32, ID[2]:ID[2]+32]
            tar_box = convert(tar_box)
        elif self.phase == 1:
            tar_box = self.ng_box[ID[0]:ID[0]+32, ID[1]:ID[1]+32, ID[2]:ID[2]+32]
        elif self.phase == 2:
            #tar_box = self.gm_box[ID[0]:ID[0]+32, ID[1]:ID[1]+32, ID[2]:ID[2]+32]
            tar_box = (self.gm_box[ID[0]:ID[0]+32, ID[1]:ID[1]+32, ID[2]:ID[2]+32]**0.25) * 1000

        if self.flipBox == True:
            dimToFlip = tuple(np.arange(3)[np.random.choice(a=[False, True], size=3)])
            if len(dimToFlip) > 0:
                inp_box = np.flip(inp_box, dimToFlip)
                tar_box = np.flip(tar_box, dimToFlip)

        inp_box = np.expand_dims(inp_box, axis=0)

        inp_box = torch.from_numpy(inp_box.copy())
        tar_box = torch.from_numpy(tar_box.copy())

        return inp_box, tar_box
