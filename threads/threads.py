from utils.model_inference import inp_model_inference, sr_model_inference, deno_model_inference, intp_model_inference
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
import json
import os

class inp_Thread(QThread):
    def __init__(self):
        super(inp_Thread, self).__init__()
        self.inp_model_name_input = ''
        print("线程收到的model:", self.inp_model_name_input)

    def run(self):
        # 取出保存的 frames 和 masks 的地址
        with open('config/file.json', 'r', encoding='utf-8') as f:
            file = json.loads(f.read())
        file_root = file['file_root']
        mask_root = file['mask_root']
        output_root = file['completed_frames_root']
        # print('目标移除的原视频帧文件路径:', file_root)
        # print('目标移除的mask文件路径:', mask_root)
        # 建立保存文件夹
        if not os.path.exists(output_root):
            os.makedirs(output_root)
        # 原图的尺寸
        input_w = file['input_w']
        input_h = file['input_h']
        # 输入 frames, mask, save_path, model_name, input_h, input_w (这2个尺寸是为了保证输入输出一致) 进行推理
        inp_model_inference(file_path=file_root, mask_path=mask_root, save_path=output_root,
                          model=self.inp_model_name_input, output_h=input_h, output_w=input_w)

        # todo 之后增加其他线程，下面 2 行代码，都要加上
        # 修复完成的信号发射，右下角的控制台显示：“已完成”
        myWin.function_completed_signal.emit()
        # 为之后 player2 的播放做准备
        myWin.completed_frames_init()