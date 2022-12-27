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
import ldm.models.autoencoder as autoencoder


if platform.system() == 'Windows':
    UTIL_DIR = r"E:\我的坚果云\sourcecode\python\util"
else:
    UTIL_DIR = r"/home/chenxu/Nutstore Files/我的坚果云/sourcecode/python/util"

sys.path.append(UTIL_DIR)
import common_metrics
import common_pelvic_pt as common_pelvic


def main(device, args):
    ckpt = os.path.join(args.log_dir, "checkpoints", "last.ckpt")
    configs = sorted(glob.glob(os.path.join(args.log_dir, "configs/*.yaml")))
    args.resume_from_checkpoint = ckpt
    args.base = configs
    model = instantiate_from_config(config.model)


    #net = torch.load(os.path.join(args.checkpoint_dir, "last.ckpt"), map_location=device)


    pdb.set_trace()
    print(111)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu', type=int, default=0, help="gpu device id")
    parser.add_argument('--data_dir', type=str, default=r'data', help='path of the dataset')
    parser.add_argument('--log_dir', type=str, default=r'checkpoints', help="checkpoint file dir")
    parser.add_argument('--output_dir', type=str, default='', help="the output directory")
    parser.add_argument("--base", nargs="*", metavar="configs/autoencoder/autoencoder_kl_pelvic.yaml",
                        help="paths to base configs. Loaded from left-to-right. "
                             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
                        default=list())

    args = parser.parse_args()

    if args.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = torch.device("cuda")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        device = torch.device("cpu")

    main(device, args)
