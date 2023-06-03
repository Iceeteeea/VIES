from models.inp_models.DSTT.test_DSTT import main_worker_DSTT
from models.inp_models.FuseFormer.test import main_worker_FuseFormer
from models.sr_models.BasicSR.inference_basicvsr import main_worker_BasicVsr
from models.deno_models.fastdvdnet.test_fastdvdnet import main_worker_fastdvdnet
from models.intp_models.RIFE.inference_video import main_worker_RIFE
from models.restore_models.remaster.remaster import main_worker_remaster
from models.color_models.SVCNet.inference_SVCNet import main_worker_SVCNet
from models.color_models.SVCNet.GCS.generate_color_scribbles_video import main_worker_color_scribbles
from models.deflicker_models.AIOD.deflicker_inference import DeflickerInference


import shutil
import os
import glob
import torch
import math


def inp_model_inference(file_path='', mask_path='', save_path='',model='',
                        output_h=0, output_w=0, inp_opt_index=[]):

    if model == 'DSTT':
       main_worker_DSTT(file_path, mask_path, save_path=save_path,
                        out_h=output_h, out_w=output_w, inp_opt_index=inp_opt_index)
    elif model == 'FuseFormer':
        main_worker_FuseFormer(file_path, mask_path, save_path=save_path, out_h=output_h, out_w=output_w)
    elif model == 'FGT':
        # todo
        pass
    else:
        print('选择了未知model')

def sr_model_inference(file_path='',  save_path='',model='',  interval=15):

    if model == 'BasicVsr':
       main_worker_BasicVsr(input_path=file_path, save_path=save_path, interval=interval)
    elif model == 'FuseFormer':
        pass

    else:
        print('选择了未知model')

def deno_model_inference(file_path='', save_path='',model='', noise_sigma=25, max_num_fr_per_seq=90):

    if model == 'fastdvdnet':
        noise_sigma = noise_sigma / 255.
        main_worker_fastdvdnet(test_path=file_path, save_path=save_path, noise_sigma=noise_sigma, model_file='models/deno_models/fastdvdnet/checkpoints/model.pth', cuda=True, gray=False, max_num_fr_per_seq=max_num_fr_per_seq)
    else:
        pass

def intp_model_inference(file_path='', save_path='', model='', rate=2, UHD=None):
    main_worker_RIFE(img_path=file_path, exp=math.log(rate, 2), output=save_path, UHD=UHD)

def res_model_inference(file_path, save_path, mindim):
    main_worker_remaster(input=file_path, outputdir=save_path, mindim=mindim)

def color_model_inference(file_path, save_path,lambda_value, sigma_color, pad='zero', scribble_root='results/color_point40_color_width5_256p', iter_frames=7, crop_size_h=256, crop_size_w=448, use_scribble=True, idx=0):

    one_color_scribble_root = os.path.join("results", "one_color_scribble")
    if not os.path.exists(one_color_scribble_root):
        os.mkdir(one_color_scribble_root)
    if not os.path.exists('results/color_point40_color_width5_256p'):
        os.mkdir('results/color_point40_color_width5_256p')
    one_color_scribble_list = sorted(glob.glob(os.path.join(one_color_scribble_root, '*.png')))


    main_worker_color_scribbles(file_path, scribble_root=one_color_scribble_list[idx],
                                save_root='results/color_point40_color_width5_256p', crop_size_w=crop_size_w, crop_size_h=crop_size_h)
    torch.cuda.empty_cache()
    main_worker_SVCNet(save_rgb_path=save_path, lambda_value=lambda_value, sigma_color=sigma_color, pad=pad, base_root=file_path, scribble_root=scribble_root, iter_frames=iter_frames,crop_size_h=crop_size_h, crop_size_w=crop_size_w, use_scribble=use_scribble)
    # shutil.rmtree('results')

def deflicker_model_inference(file_path, save_path):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    os.environ['MKL_SERVICE_FORCE_INTEL'] = 1

    DI = DeflickerInference(file_path, "VIES")
    DI.inference()
    shutil.copytree("results/frames/final/output", save_path)
    shutil.rmtree('results')
    shutil.rmtree('data/test')

def deblur_model_inference(file_path, save_path):
    DI = DeflickerInference(file_path, "VIES")
    DI.environ()
    end = glob.glob(os.path.join(file_path, '*'))[0]
    end = os.path.splitext(end)[-1]
    end = '{:05d}' + end
    # print(end)
    deblur_cmd = "python models/deblur_model/FGST/restoration_video_demo.py models/deblur_model/FGST/configs/FGST_deblur_dvd_test.py models/deblur_model/FGST/checkpoints/FGST_dvd.pth {} {} --filename-tmpl {}".format(file_path, save_path, end)
    # print(deblur_cmd)
    os.system(deblur_cmd)

if __name__ == '__main__':
    deflicker_model_inference("data/inpainting/frames", "test22")
    # deblur_model_inference(file_path="data/inpainting/frames", save_path="test22")