# -*- coding: utf-8 -*-

import argparse
import os
import sys
import pdb
import torch
import time
import numpy
import platform
import skimage.io
import glob
import torch
import main_pelvic

if platform.system() == 'Windows':
    UTIL_DIR = r"E:\我的坚果云\sourcecode\python\util"
else:
    UTIL_DIR = r"/home/chenxu/Nutstore Files/我的坚果云/sourcecode/python/util"

sys.path.append(UTIL_DIR)
import common_pelvic_pt as common_pelvic


def main():
    dataset = main_pelvic.PelvicDatasetEx(r"D:\datasets\pelvic\h5_data_nonrigid", "ct", n_slices=3)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    for data in dataloader:
        #ori_image = data["ori_image"].detach().cpu().numpy()
        image = data["image"].detach().cpu().numpy()

        print(image.shape)
        
        #ori_image = common_pelvic.generate_display_image(ori_image, is_seg=False)
        image = common_pelvic.generate_display_image(image, is_seg=False)
        
        #skimage.io.imsave(os.path.join("outputs", "ori_im.jpg"), ori_image)
        skimage.io.imsave(os.path.join("outputs", "im.jpg"), image)

        print("xxxx")
        break


if __name__ == '__main__':
    main()
