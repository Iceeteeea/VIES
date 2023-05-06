import glob
import json
import os
import cv2

def choose_outroot():
    with open('config/file.json', 'r', encoding='utf-8') as f:
        file = json.loads(f.read())
    current_mode = file['current_mode']
    if current_mode == '目标移除':
        return file['inp_completed_frames_root']
    elif current_mode == '威亚移除':
        pass
    elif current_mode == '去噪':
        pass
    elif current_mode == '去无损':
        pass
    elif current_mode == '超分辨率':
        return file['sr_completed_frames_root']
    elif current_mode == '帧率增强':
        pass

def write_config(key:str, value):
    config_file = 'config/file.json'
    config = json.load(open(config_file, 'r', encoding='utf-8'))
    config[key] = value
    config_json = json.dumps(config, ensure_ascii=False, indent=2)
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(config_json)

def init_config():
    config_file = 'config/file.json'
    config = json.load(open(config_file, 'r', encoding='utf-8'))
    config["interpolation_rate"] = 2
    config["deno_strength"] = 25
    config["mindim"] = 320
    config["lambda_value"] = 500
    config["sigma_color"] = 4
    config["iter_frames"] = 7
    config["crop_size_h"] = 256
    config["crop_size_w"] = 448
    config["color_mode"] = 1
    config["UHD"] = False
    config["color_image_idx"] = 0 # 上色中显示的颜色参考图的索引
    config_json = json.dumps(config, ensure_ascii=False, indent=2)
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(config_json)

def read_config(key:str):
    with open('config/file.json', 'r', encoding='utf-8') as f:
        file = json.loads(f.read())
    return file[key]

def img2video(save_path):
    with open('config/file.json', 'r', encoding='utf-8') as f:
        file = json.loads(f.read())
    w = file["input_w"]
    h = file["input_h"]
    fps = file['interpolation_rate'] * 25
    imgs_path = file["completed_frames_root"]
    size = (w, h)
    videoWrite = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    imgs_lst = sorted(glob.glob(os.path.join(imgs_path, '*.png')))

    for img in imgs_lst:
        frame = cv2.imread(img)
        videoWrite.write(frame)
    videoWrite.release()
    print('done')