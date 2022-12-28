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
#import ldm.models.autoencoder as autoencoder
from ldm.models.diffusion.ddpm import LatentDiffusion
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

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with torch.no_grad():
        z_channels = config.model.params.unet_config.params.in_channels
        image_size = config.model.params.unet_config.params.image_size
        z = model.p_sample_loop(None, (16, z_channels, image_size, image_size))
        syn_im = model.decode_first_stage(z)
        syn_im = syn_im.detach().cpu().numpy().clip(-1, 1)

        syn_im = common_pelvic.generate_display_image(syn_im)
        skimage.io.imsave(os.path.join(args.output_dir, "syn_im.jpg"), syn_im)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu', type=int, default=0, help="gpu device id")
    parser.add_argument('--data_dir', type=str, default=r'/home/chenxu/datasets/pelvic/h5_data_nonrigid/', help='path of the dataset')
    parser.add_argument('--log_dir', type=str, default=r'/home/chenxu/training/logs/ldm/ct_kl/2022-12-27T19-48-14_pelvic/', help="checkpoint file dir")
    parser.add_argument('--output_dir', type=str, default='/home/chenxu/training/test_output/ldm/ct_kl', help="the output directory")
    parser.add_argument("--base", nargs="*", metavar="configs/latent-diffusion/pelvic-vq-f8.yaml",
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
