import json
import os

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
    config_json = json.dumps(config, ensure_ascii=False, indent=2)
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(config_json)

def read_config(key:str):
    with open('config/file.json', 'r', encoding='utf-8') as f:
        file = json.loads(f.read())
    return file[key]