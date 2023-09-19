import os
import cv2
import numpy as np
import imageio
import argparse
import torch


def make_gif(images,path, duration = 0.04):
    """
    images: [B,T,W,H]
    """
    if isinstance(images, list):
        read_path = isinstance(images[0],str)
        time_step = len(images)
    else:
        time_step,W,H,C = images.shape
    frames = []
    for i in range(time_step):
        if read_path:
            frame = cv2.imread(images[i])
        else:
            frame = images[i]
        frames.append(frame)
    imageio.mimsave(path, images, duration = duration)