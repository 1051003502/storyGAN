import torch.backends.cudnn as cudnn
import torch
import torchvision.transforms as transforms
import PIL
import argparse
import os
import random
import sys
import pprint
import dateutil
import dateutil.tz
import numpy as np
import functools
import clevr_data as data
import pdb
from miscc.config import cfg, cfg_from_file
if __name__ == '__main__':
    n_channels = 3
    dir_path = '../clevr_dataset/'
    image_transforms = transforms.Compose([
        PIL.Image.fromarray,
        transforms.Resize((cfg.IMSIZE, cfg.IMSIZE)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        lambda x: x[:n_channels, ::],
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    imagedataset = data.ImageDataset(dir_path, image_transforms, cfg.VIDEO_LEN, True)
    data0=imagedataset[2]
    print('')