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

    test_data, _, _, _ = common_pelvic.load_test_data(args.data_dir)
    patch_shape = (1, test_data.shape[2], test_data.shape[3])

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

    """
        syn_img, codes = model.forward(torch.tensor(test_img, device=device))
    
    syn_img = torch.clamp(syn_img, -1, 1)
    syn_img = syn_img.detach().cpu().numpy()
    pdb.set_trace()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

        for i in range(syn_img.shape[0]):
            skimage.io.imsave(os.path.join(args.output_dir, "ori_%d.jpg" % i), common_pelvic.data_restore(test_img[i, 0, :, :]))
            skimage.io.imsave(os.path.join(args.output_dir, "syn_%d.jpg" % i), common_pelvic.data_restore(syn_img[i, 0, :, :]))
    """

    print("psnr:%f/%f" % (psnr_list.mean(), psnr_list.std()))


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