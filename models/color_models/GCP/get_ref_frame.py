from models.color_models.GCP.predict_imagenet_label import predict_label
from models.color_models.GCP.main import color

from models.color_models.SVCNet.GCS.generate_color_scribbles_ImageNet import get_one_color_scribble
import glob
import os
import shutil
import torch
from utils.utils import read_config


def get_ref_frame(input: str):
    test_frame = sorted(glob.glob(os.path.join(input, '*')))[0]
    if not os.path.exists('tmp'):
        os.mkdir('tmp')
    shutil.copy(test_frame, 'tmp')
    predict_label(data='tmp')  # 给输入帧上的物体分类，以便后续上色
    color(test_folder=test_frame)
    shutil.rmtree("tmp")
    one_color_scribble_root = os.path.join("results", "one_color_scribble")
    crop_size_h = read_config("crop_size_h")
    crop_size_w = read_config("crop_size_w")
    get_one_color_scribble(baseroot='results/inference_random_diverse_color/full_resolution_results',
                           saveroot=one_color_scribble_root, crop_size_w=crop_size_w, crop_size_h=crop_size_h)


    torch.cuda.empty_cache()
