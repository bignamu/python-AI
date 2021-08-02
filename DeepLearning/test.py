import datetime
import os
import time

import torch
import torch.utils.data
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn

import cv2
import numpy as np

import argparse


parser = argparse.ArgumentParser(description='PyTorch Detection Training')

parser.add_argument('--data-path', default='D:ExDark', help='dataset')
parser.add_argument('--dataset', default='voc', help='dataset')

args = parser.parse_args()

print(args)
print(args.data_path)