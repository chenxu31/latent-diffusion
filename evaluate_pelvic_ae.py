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
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config


if platform.system() == 'Windows':
    UTIL_DIR = r"E:\我的坚果云\sourcecode\python\util"
else:
    UTIL_DIR = r"/home/chenxu/Nutstore Files/我的坚果云/sourcecode/python/util"

sys.path.append(UTIL_DIR)
import common_metrics
import common_pelvic_pt as common_pelvic
import common_net_pt as common_net


def main(device, args):
    ckpt_file = os.path.join(args.log_dir, "checkpoints", "last.ckpt")
    
    config_files = sorted(glob.glob(os.path.join(args.log_dir, "configs/*.yaml")))
    configs = [OmegaConf.load(cfg) for cfg in config_files]
    #cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs)#, cli)
    model = instantiate_from_config(config.model)
    model.init_from_ckpt(ckpt_file)
    model.to(device)
    model.eval()

    if args.modality == "ct":
        test_data, _, _, _ = common_pelvic.load_test_data(args.data_dir)
    elif args.modality == "cbct":
        _, test_data, _, _ = common_pelvic.load_test_data(args.data_dir)
    else:
        assert 0
    patch_shape = (config.model.params.ddconfig.in_channels, test_data.shape[2], test_data.shape[3])

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    psnr_list = numpy.zeros((test_data.shape[0],), numpy.float32)
    with torch.no_grad():
        for i in range(test_data.shape[0]):
            syn_im = common_net.produce_results(device, model, [patch_shape, ], [test_data[i], ],
                                                data_shape=test_data.shape[1:], patch_shape=patch_shape, is_seg=False,
                                                batch_size=16)

            syn_im = syn_im.clip(-1, 1)
            psnr_list[i] = common_metrics.psnr(syn_im, test_data[i])

            if args.output_dir:
                common_pelvic.save_nii(syn_im, os.path.join(args.output_dir, "syn_%d.nii.gz" % i))

    msg = "psnr:%f/%f" % (psnr_list.mean(), psnr_list.std())
    print(msg)

    with open(os.path.join(args.output_dir, "results.txt"), "w") as f:
        f.write(msg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu', type=int, default=0, help="gpu device id")
    parser.add_argument('--data_dir', type=str, default=r'/home/chenxu/datasets/pelvic/h5_data_nonrigid/', help='path of the dataset')
    parser.add_argument('--log_dir', type=str, default=r'checkpoints', help="checkpoint file dir")
    parser.add_argument('--output_dir', type=str, default='', help="the output directory")
    parser.add_argument('--modality', type=str, default='ct', choices=["ct", "cbct"], help="the output directory")

    args = parser.parse_args()

    if args.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = torch.device("cuda")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        device = torch.device("cpu")

    main(device, args)
