# -*- coding: utf-8 -*-
import cv2
from PIL import Image
import numpy as np
import importlib
import os
import argparse
import torch
from torchvision import transforms

# from core.utils import Stack, ToTorchFormatTensor
from models.inp_models.DSTT.core.utils import Stack, ToTorchFormatTensor

parser = argparse.ArgumentParser(description="DSTT")
parser.add_argument("-v", "--video", type=str, default='')
parser.add_argument("-m", "--mask",   type=str, default='')
parser.add_argument("-c", "--ckpt",   type=str, default='models/inp_models/DSTT/checkpoints/dstt.pth')
parser.add_argument("--model", type=str, default='DSTT')
parser.add_argument("--width", type=int, default=432)
parser.add_argument("--height", type=int, default=240)
parser.add_argument("--outw", type=int, default=432)
parser.add_argument("--outh", type=int, default=240)
parser.add_argument("--step", type=int, default=10)
parser.add_argument("--num_ref", type=int, default=-1)
parser.add_argument("--neighbor_stride", type=int, default=5)
parser.add_argument("--savefps", type=int, default=24)
parser.add_argument("--use_mp4", action='store_true')
args = parser.parse_args()


w, h = args.width, args.height
ref_length = args.step  # ref_step
num_ref = args.num_ref
neighbor_stride = args.neighbor_stride
default_fps = args.savefps

_to_tensors = transforms.Compose([
    Stack(),
    ToTorchFormatTensor()])


# sample reference frames from the whole video 
def get_ref_index(f, neighbor_ids, length):
    ref_index = []
    if num_ref == -1:
        for i in range(0, length, ref_length):
            if not i in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, f - ref_length * (num_ref//2))
        end_idx = min(length, f + ref_length * (num_ref//2))
        for i in range(start_idx, end_idx+1, ref_length):
            if not i in neighbor_ids:
                if len(ref_index) > num_ref:
                # if len(ref_index) >= 5-len(neighbor_ids):
                    break
                ref_index.append(i)
    return ref_index


# read frame-wise masks
# def read_mask(mpath):
#     masks = []
#     mnames = os.listdir(mpath)
#     mnames.sort()
#     for m in mnames:
#         m = Image.open(os.path.join(mpath, m))
#         m = m.resize((w, h), Image.NEAREST)
#         m = np.array(m.convert('L'))
#         m = np.array(m > 0).astype(np.uint8)
#         m = cv2.dilate(m, cv2.getStructuringElement(
#             cv2.MORPH_CROSS, (3, 3)), iterations=4)
#         masks.append(Image.fromarray(m*255))
#     return masks

def read_mask(mpath, opti_frames_index):
    index = opti_frames_index
    print('mask接收的优化index:', index)
    masks = []
    mnames = os.listdir(mpath)
    mnames.sort()
    # for m in mnames:
    for i, m in enumerate(mnames):
        m = Image.open(os.path.join(mpath, m))
        m = m.resize((w, h), Image.NEAREST)
        m = np.array(m.convert('L'))
        m = np.array(m > 0).astype(np.uint8)
        m = cv2.dilate(m, cv2.getStructuringElement(
            cv2.MORPH_CROSS, (3, 3)), iterations=4)

        if index == []:
            pass
        else:
            if i in index:
                # 这是需要优化的帧，mask不变
                pass
            else:
                # 这是不需要优化的帧，mask全部为 0 (无掩膜区域)
                m[:, :] = 0

        masks.append(Image.fromarray(m*255))
    return masks



#  read frames from video 
def read_frame_from_videos(args, file_path):
    # vname = args.video
    vname = file_path
    frames = []
    if args.use_mp4:
        vidcap = cv2.VideoCapture(vname)
        success, image = vidcap.read()
        count = 0
        while success:
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            frames.append(image.resize((w, h)))
            success, image = vidcap.read()
            count += 1
    else:
        lst = os.listdir(vname)
        lst.sort()
        fr_lst = [vname+'/'+name for name in lst]
        for fr in fr_lst:
            image = cv2.imread(fr)
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            frames.append(image.resize((w, h)))
    return frames       


def main_worker_DSTT(file_path='', mask_path='', save_path='', out_h=0, out_w=0, inp_opt_index=[]):
    # set up models 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = importlib.import_module('models.inp_models.DSTT.model.' + args.model)
    model = net.InpaintGenerator().to(device)
    data = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(data['netG'])
    print('loading from: {}'.format(args.ckpt))
    model.eval()

    # prepare datset, encode all frames into deep space 
    frames = read_frame_from_videos(args, file_path)
    video_length = len(frames)
    imgs = _to_tensors(frames).unsqueeze(0)*2-1
    frames = [np.array(f).astype(np.uint8) for f in frames]

    # masks = read_mask(args.mask)
    masks = read_mask(mask_path, inp_opt_index)
    binary_masks = [np.expand_dims((np.array(m) != 0).astype(np.uint8), 2) for m in masks]
    masks = _to_tensors(masks).unsqueeze(0)
    imgs, masks = imgs.to(device), masks.to(device)
    comp_frames = [None]*video_length
    # print('loading videos and masks from: {}'.format(args.video))
    print('*** DSTT模型开始修复 ***')
    # completing holes by spatial-temporal transformers
    for f in range(0, video_length, neighbor_stride):
        neighbor_ids = [i for i in range(max(0, f-neighbor_stride), min(video_length, f+neighbor_stride+1))]
        ref_ids = get_ref_index(f, neighbor_ids, video_length)
        # print(f, len(neighbor_ids), len(ref_ids))
        len_temp = len(neighbor_ids) + len(ref_ids)
        selected_imgs = imgs[:1, neighbor_ids+ref_ids, :, :, :]
        selected_masks = masks[:1, neighbor_ids+ref_ids, :, :, :]
        with torch.no_grad():
            masked_imgs = selected_imgs*(1-selected_masks)
            pred_img = model(masked_imgs)
            pred_img = (pred_img + 1) / 2
            pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy()*255
            for i in range(len(neighbor_ids)):
                idx = neighbor_ids[i]
                img = np.array(pred_img[i]).astype(
                    np.uint8)*binary_masks[idx] + frames[idx] * (1-binary_masks[idx])
                if comp_frames[idx] is None:
                    comp_frames[idx] = img
                else:
                    comp_frames[idx] = comp_frames[idx].astype(
                        np.float32)*0.5 + img.astype(np.float32)*0.5

    # 以图片的形式存储
    for f in range(video_length):
        comp = comp_frames[f]
        comp = cv2.cvtColor(np.array(comp), cv2.COLOR_BGR2RGB)
        # if w != out_w:
        comp = cv2.resize(comp, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(save_path, "{:04}.png".format(f)), comp)

        if f == video_length-1:
            print('保存完成')


    # # aa, bb, cc = args.ckpt.split('/')
    # a1, a2, bb, cc = args.ckpt.split('/')
    # name = bb + '_' + cc.split('.')[0] + '_' + args.mask.split('/')[-1]
    # writer = cv2.VideoWriter(f"cache/{name}_result.mp4", cv2.VideoWriter_fourcc(*"mp4v"), default_fps, (args.outw, args.outh))
    # for f in range(video_length):
    #     comp = np.array(comp_frames[f]).astype(
    #         np.uint8)*binary_masks[f] + frames[f] * (1-binary_masks[f])
    #     if w != args.outw:
    #         comp = cv2.resize(comp, (args.outw, args.outh), interpolation=cv2.INTER_LINEAR)
    #     writer.write(cv2.cvtColor(np.array(comp).astype(np.uint8), cv2.COLOR_BGR2RGB))
    # writer.release()
    # print('Finish in {}'.format(f"cache/{name}_result_DSTT.mp4"))
    # save_path = '修复后文件的保存路径：***'
    #
    # return save_path

# if __name__ == '__main__':
#     model = main_worker_DSTT(file_path='/home/will/space2/_92_gui/_04_code/_02_VI/examples/tennis/tennis',
#                              mask_path='/home/will/space2/_92_gui/_04_code/_02_VI/examples/tennis/tennis_mask')
#     print(model)