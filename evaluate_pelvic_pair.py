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


@torch.no_grad()
def produce_results(model, con_data, patch_shape, batch_size=16):
    con_input = numpy.zeros((batch_size, patch_shape[0], con_data.shape[1], con_data.shape[2]), numpy.float32)
    syn_data = numpy.zeros_like(con_data)
    used = numpy.zeros_like(con_data)

    idx = 0
    batch_locs = [None] * batch_size
    for i in range(con_data.shape[0]):
        batch_locs[idx] = i
        con_input[idx, :, :, :] = con_data[i:i + patch_shape[0], :, :]

        idx += 1
        if idx < batch_size:
            continue

        idx = 0

        con_posterior = model.encode_first_stage(torch.tensor(con_input, device=model.device))
        con_z = model.get_first_stage_encoding(con_posterior).detach()
        syn_z = model.p_sample_loop(con_z, con_z.shape)
        syn_im = model.decode_first_stage(syn_z)
        syn_im = syn_im.detach().cpu().numpy().clip(-1, 1)

        for batch_id, loc in enumerate(batch_locs):
            syn_data[loc:loc + patch_shape[0], :, :] += syn_im[batch_id, :, :, :]
            used[loc:loc + patch_shape[0], :, :] += 1.

    if idx != 0:
        con_posterior = model.encode_first_stage(torch.tensor(con_input, device=model.device))
        con_z = model.get_first_stage_encoding(con_posterior).detach()
        syn_z = model.p_sample_loop(con_z, con_z.shape)
        syn_im = model.decode_first_stage(syn_z)
        syn_im = syn_im.detach().cpu().numpy().clip(-1, 1)

        for batch_id, loc in enumerate(batch_locs[:idx]):
            syn_data[loc:loc + patch_shape[0], :, :] += syn_im[batch_id, :, :, :]
            used[loc:loc + patch_shape[0], :, :] += 1.

    for _used in used:
        assert _used.min() > 0

    syn_data /= used

    return syn_data


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

    ct_data, cbct_data, _, _ = common_pelvic.load_val_data(args.data_dir)
    patch_depth = config.model.params.first_stage_config.params.ddconfig.in_channels
    patch_shape = (patch_depth, ct_data.shape[2], ct_data.shape[3])
    
    ct_data = ct_data[:1, 100:100+16, :, :]
    cbct_data = cbct_data[:1, 100:100+16, :, :]

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    psnr_list = numpy.zeros((ct_data.shape[0],), numpy.float32)
    with torch.no_grad():
        for i in range(cbct_data.shape[0]):
            syn_data = produce_results(model, cbct_data[i], patch_shape=patch_shape, batch_size=16)

	    
            #syn_data = syn_data.clip(-1, 1)
            #psnr_list[i] = common_metrics.psnr(syn_data, ct_data[i])

            if args.output_dir:
                image = numpy.concatenate([ct_data, cbct_data, numpy.expand_dims(syn_data, 0)], 3)
                image = common_pelvic.generate_display_image(image)
                skimage.io.imsave(os.path.join(args.output_dir, "syn_im.jpg"), image)
                #common_pelvic.save_nii(syn_data, os.path.join(args.output_dir, "syn_%d.nii.gz" % i))

    print("psnr:%f/%f" % (psnr_list.mean(), psnr_list.std()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu', type=int, default=0, help="gpu device id")
    parser.add_argument('--data_dir', type=str, default=r'/home/chenxu/datasets/pelvic/h5_data_nonrigid/', help='path of the dataset')
    parser.add_argument('--log_dir', type=str, default=r'/home/chenxu/training/logs/ldm/ct_kl_pair/2022-12-28T11-15-29_pelvic/', help="checkpoint file dir")
    parser.add_argument('--output_dir', type=str, default='/home/chenxu/training/test_output/ldm/ct_kl_pair', help="the output directory")
    parser.add_argument("--base", nargs="*", metavar="configs/latent-diffusion/pelvic-vq-f8_pair.yaml",
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
