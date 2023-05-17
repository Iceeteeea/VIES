import os.path

from PyQt5.QtWidgets import QMainWindow, QDialog, QMessageBox
from PyQt5.QtWidgets import (QApplication, QSizePolicy, QSlider,
                             QShortcut, QFileDialog)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QKeySequence, QImage, QTextCursor
from PyQt5.QtGui import QIcon
from PyQt5 import QtCore

import sys
import json
from argparse import ArgumentParser
from os import path
import functools
import shutil

# ui 转为 python 命令：python -m PyQt5.uic.pyuic main_win_v1.ui -o main_win_v1.py
from main_win.win_v2.main_win_v10_1 import Ui_MainWindow
from main_win.win_v2.sub1_win_v8 import Ui_Dialog_tool
from main_win.win_v2.sub2_win_v11 import Ui_Dialog_Mask
# 视频修复的相关模型
from utils.model_inference import inp_model_inference, sr_model_inference, deno_model_inference, intp_model_inference, \
    res_model_inference, color_model_inference
from models.color_models.GCP.get_ref_frame import get_ref_frame

from mask_gen_model.inference_core import InferenceCore
from mask_gen_model.interact.s2m_controller import S2MController
from mask_gen_model.interact.fbrs_controller import FBRSController
from mask_gen_model.model.propagation.prop_net import PropagationNetwork
from mask_gen_model.model.fusion_net import FusionNet
from mask_gen_model.model.s2m.s2m_network import deeplabv3plus_resnet50 as S2M
from mask_gen_model.util.tensor_util import unpad_3dim
from mask_gen_model.util.palette import pal_color_map

from mask_gen_model.interact.interactive_utils import *
from mask_gen_model.interact.interaction import *
from mask_gen_model.interact.timer import Timer
from utils.utils import choose_outroot, write_config, init_config, read_config, img2video
torch.set_grad_enabled(False)
# DAVIS palette
palette = pal_color_map()


# todo 2023.4.11 --> 插帧功能实现后，视频的播放需要注意，因为帧数不相同

# todo  注意：在增加 超分、插帧功能后，需要注意 result_save_video()函数，因为里面涉及保存时的帧率和尺寸大小。
# todo  在打开原视频的时候，已经将 输入尺寸和fps 存入了json中。可以将原尺寸与放大倍数相乘，作为最后的输出尺寸。

# 修复线程
class inp_Thread(QThread):
    def __init__(self):
        super(inp_Thread, self).__init__()
        self.inp_model_name_input = ''
        print("线程收到的model:", self.inp_model_name_input)
        # 是否为优化模式
        self.inp_optimization_flag = 0
        self.inp_optimization_index = []

    def run(self):
        # 取出保存的 frames 和 masks 的地址
        with open(os.path.join('config', 'file.json'), 'r', encoding='utf-8') as f:
            file = json.loads(f.read())
        # print('self.inp_optimization_flag', self.inp_optimization_flag)
        if self.inp_optimization_flag == 0:
            # file_root = file['file_root']
            if myWin.follow_flag == 0:
                file_root = file['file_root']
            else:
                file_root = file['follow_root']
            output_root = file['completed_frames_root']
        else:
            print("优化模式--1")
            file_root = file['completed_frames_root']
            output_root = file['inp_opti_completed_frames_root']
        # file_root = file['file_root']
        # output_root = file['completed_frames_root']
        mask_root = file['mask_root']
        # output_root = file['completed_frames_root']

        # 建立保存文件夹
        if not os.path.exists(output_root):
            os.makedirs(output_root)
        # 原图的尺寸
        input_w = file['input_w']
        input_h = file['input_h']
        print('self.inp_optimization_index *:', self.inp_optimization_index)
        # 输入 frames, mask, save_path, model_name, input_h, input_w (这2个尺寸是为了保证输入输出一致) 进行推理
        inp_model_inference(file_path=file_root, mask_path=mask_root, save_path=output_root,
                          model=self.inp_model_name_input, output_h=input_h, output_w=input_w,
                            inp_opt_index=self.inp_optimization_index)

        # todo 之后增加其他线程，下面 2 行代码，都要加上
        # 修复完成的信号发射，右下角的控制台显示：“已完成”
        myWin.function_completed_signal.emit()
        # 为之后 player2 的播放做准备
        myWin.completed_frames_init()


# todo 去噪声
# 去噪线程
class denoising_Thread(QThread):
    def __init__(self):
        super(denoising_Thread, self).__init__()
        self.denoising_model_name_input = ''
        print("线程收到的model:", self.denoising_model_name_input)

    def run(self):
        # 取出保存的 frames
        with open(os.path.join('config', 'file.json'), 'r', encoding='utf-8') as f:
            file = json.loads(f.read())
        if myWin.follow_flag == 0:
            file_root = file['file_root']
        else:
            file_root = file['follow_root']
        output_root = file['completed_frames_root']
        deno_strength = file['deno_strength']
        max_num_fr_per_seq = file["max_num_fr_per_seq"]
        # 建立保存文件夹
        if not os.path.exists(output_root):
            os.makedirs(output_root)

        deno_model_inference(file_path=file_root,  save_path=output_root,
                          model=self.denoising_model_name_input, noise_sigma=deno_strength, max_num_fr_per_seq=max_num_fr_per_seq)

        # todo 之后增加其他线程，下面 2 行代码，都要加上
        # 修复完成的信号发射，右下角的控制台显示：“已完成”
        myWin.function_completed_signal.emit()
        # 为之后 player2 的播放做准备
        myWin.completed_frames_init()


# 超分线程
class sr_Thread(QThread):
    def __init__(self):
        super(sr_Thread, self).__init__()
        self.sr_model_name_input = ''
        print("线程收到的model:", self.sr_model_name_input)

    def run(self):
        # 取出保存的 frames
        with open(os.path.join('config', 'file.json'), 'r', encoding='utf-8') as f:
            file = json.loads(f.read())
        # file_root = file['file_root']
        if myWin.follow_flag == 0:
            file_root = file['file_root']
        else:
            file_root = file['follow_root']
        output_root = file['completed_frames_root']
        # 建立保存文件夹
        if not os.path.exists(output_root):
            os.makedirs(output_root)
        sr_model_inference(file_path=file_root,  save_path=output_root,
                          model=self.sr_model_name_input)
        # todo 之后增加其他线程，下面 2 行代码，都要加上
        # 修复完成的信号发射，右下角的控制台显示：“已完成”
        myWin.function_completed_signal.emit()
        # 为之后 player2 的播放做准备
        myWin.completed_frames_init()


# 插帧线程
class intp_Thread(QThread):
    def __init__(self):
        super(intp_Thread, self).__init__()
        self.intp_model_name_input = ''
        print("线程收到的model:", self.intp_model_name_input)
        self.intp_rate = 2

    def run(self):
        # 取出保存的 frames
        with open(os.path.join('config', 'file.json'), 'r', encoding='utf-8') as f:
            file = json.loads(f.read())
        # file_root = file['file_root']
        if myWin.follow_flag == 0:
            file_root = file['file_root']
        else:
            file_root = file['follow_root']
        output_root = file['completed_frames_root']
        rate = file["interpolation_rate"]
        UHD = file["UHD"]
        # 用于定时器2,来计算播放速率
        myWin.intp_rate_timer_2_2 = rate

        # 建立保存文件夹
        if not os.path.exists(output_root):
            os.makedirs(output_root)

        intp_model_inference(file_path=file_root,  save_path=output_root,rate=rate, UHD=UHD)

        # todo 之后增加其他线程，下面 2 行代码，都要加上
        # 修复完成的信号发射，右下角的控制台显示：“已完成”
        myWin.function_completed_signal.emit()
        # 为之后 player2 的播放做准备
        myWin.completed_frames_init()

# 去污线程
class restore_Thread(QThread):
    def __init__(self):
        super(restore_Thread, self).__init__()
        self.res_model_name_input = ''
        print("线程收到的model:", self.res_model_name_input)

    def run(self):
        # 取出保存的 frames
        with open(os.path.join('config', 'file.json'), 'r', encoding='utf-8') as f:
            file = json.loads(f.read())
        # file_root = file['file_root']
        if myWin.follow_flag == 0:
            file_root = file['file_root']
        else:
            file_root = file['follow_root']
        output_root = file['completed_frames_root']
        mindim = file["mindim"]
        # 建立保存文件夹
        if not os.path.exists(output_root):
            os.makedirs(output_root)

        res_model_inference(file_path=file_root,  save_path=output_root,
                          mindim=mindim)
        # todo 之后增加其他线程，下面 2 行代码，都要加上
        # 修复完成的信号发射，右下角的控制台显示：“已完成”
        myWin.function_completed_signal.emit()
        # 为之后 player2 的播放做准备
        myWin.completed_frames_init()

# 上色线程
class colorization_Thread(QThread):
    def __init__(self):
        super(colorization_Thread, self).__init__()
        self.color_model_name_input = ''
        print("线程收到的model:", self.color_model_name_input)

    def run(self):
        # 取出保存的 frames
        with open(os.path.join('config', 'file.json'), 'r', encoding='utf-8') as f:
            file = json.loads(f.read())
        # file_root = file['file_root']
        if myWin.follow_flag == 0:
            file_root = file['file_root']
        else:
            file_root = file['follow_root']
        output_root = file['completed_frames_root']
        lambda_value = file['lambda_value']
        sigma_color = file['sigma_color']
        iter_frames = file['iter_frames']
        crop_size_h = file['crop_size_h']
        crop_size_w = file['crop_size_w']
        color_image_idx = file['color_image_idx']

        # 建立保存文件夹
        if not os.path.exists(output_root):
            os.makedirs(output_root)

        # try:
        color_model_inference(file_path=file_root, save_path=output_root, lambda_value=lambda_value, sigma_color=sigma_color, iter_frames=iter_frames, crop_size_h=crop_size_h, crop_size_w=crop_size_w, idx=color_image_idx)
        # except:
        #     myWin.color_error_signal.emit() # 如果在点了颜色参考前开始上色，则触发信号报错。在控制台显示先进行颜色参考的生成

        # todo 之后增加其他线程，下面 2 行代码，都要加上
        # 修复完成的信号发射，右下角的控制台显示：“已完成”
        myWin.function_completed_signal.emit()
        # 为之后 player2 的播放做准备
        myWin.completed_frames_init()
# mask生成的初始化线程
class mask_gen_init_Thread(QThread):
    def __init__(self):
        super(mask_gen_init_Thread, self).__init__()

    def run(self):

        parser = ArgumentParser()
        parser.add_argument('--prop_model', default='mask_gen_model/saves/propagation_model.pth')
        parser.add_argument('--fusion_model', default='mask_gen_model/saves/fusion.pth')
        parser.add_argument('--s2m_model', default='mask_gen_model/saves/s2m.pth')
        parser.add_argument('--fbrs_model', default='mask_gen_model/saves/fbrs.pth')
        # (1)-1
        parser.add_argument('--images', default='',
                            help='Folder containing input images. Either this or --video needs to be specified.')
        # (1)-2
        parser.add_argument('--video',
                            help='Video file readable by OpenCV. Either this or --images needs to be specified.',
                            default='')
        parser.add_argument('--num_objects', help='Default: 1 if no masks provided, masks.max() otherwise', type=int)
        parser.add_argument('--mem_freq', default=5, type=int)
        parser.add_argument('--mem_profile', default=0, type=int,
                            help='0 - Faster and more memory intensive; 2 - Slower and less memory intensive. Default: 0.')
        parser.add_argument('--masks', help='Optional, ground truth masks', default=None)
        parser.add_argument('--no_amp', help='Turn off AMP', action='store_true')
        parser.add_argument('--resolution', help='Pass -1 to use original size', default=480, type=int)
        args = parser.parse_args()
        # 输入的原图的路径
        # with open(os.path.join('config', 'file.json', 'r', encoding='utf-8') as f:
        #     file = json.loads(f.read())
        # args.images = file['file_root']

        with torch.cuda.amp.autocast(enabled=not args.no_amp):
            # Load our checkpoint   加载传播模型
            prop_saved = torch.load(args.prop_model)
            prop_model = PropagationNetwork().cuda().eval()
            prop_model.load_state_dict(prop_saved)
            # 加载融合模型
            fusion_saved = torch.load(args.fusion_model)
            fusion_model = FusionNet().cuda().eval()
            fusion_model.load_state_dict(fusion_saved)

            # Loads the S2M model   “涂鸦”的方式，生成单帧中目标的mask的模型
            if args.s2m_model is not None:
                s2m_saved = torch.load(args.s2m_model)
                s2m_model = S2M().cuda().eval()
                s2m_model.load_state_dict(s2m_saved)
            else:
                s2m_model = None

            # Loads the images/masks, 返回固定尺寸的(t, ... )， np.array
            # if args.images is not None:
            #     images = load_images(args.images, args.resolution if args.resolution > 0 else None)
            # elif args.video is not None:
            #     images = load_video(args.video, args.resolution if args.resolution > 0 else None)
            # else:
            #     raise NotImplementedError('You must specify either --images or --video!')

            if args.masks is not None:
                masks = load_masks(args.masks)
            else:
                masks = None

            # Determine the number of objects
            # num_objects = args.num_objects
            num_objects = 1
            if num_objects is None:
                if masks is not None:
                    num_objects = masks.max()
                    print('1: num_objects:', num_objects)
                else:
                    num_objects = 1
            # num_objects = 1

            s2m_controller = S2MController(s2m_model, num_objects, ignore_class=255)

            # "点击"的方式，生成单帧中目标的mask的模型
            if args.fbrs_model is not None:
                fbrs_controller = FBRSController(args.fbrs_model)
            else:
                fbrs_controller = None

        # return prop_model, fusion_model, s2m_controller, fbrs_controller, \
        #        images, masks, num_objects, args.mem_freq, args.mem_profile
        return prop_model, fusion_model, s2m_controller, fbrs_controller, \
               masks, num_objects, args.mem_freq, args.mem_profile


class MainWindow(QMainWindow, Ui_MainWindow):
    # 信号
    quit_signal = pyqtSignal(str)
    pervious_frame_signal = pyqtSignal(str)
    next_frame_signal = pyqtSignal()
    to_SN_Slider_signal = pyqtSignal(str)
    SN_Slider_to_others_signal = pyqtSignal()
    open_file_player_init_signal = pyqtSignal()
    # 修复子窗口的修复状态改变的信号
    function_completed_signal = pyqtSignal()
    color_completed_signal = pyqtSignal()
    color_error_signal = pyqtSignal()

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)

        # 测试使用
        # self.splitter.setDisabled(True)

        # 完成了什么功能，什么功能的 flag = 1,其他为0；初始时全部为 0
        self.inp_deno_deg_sr_inter_deold_stab_flags = [0, 0, 0, 0, 0, 0, 0]
        # 当前帧的索引号，用来显示
        self.index = -1
        self.index_2 = -1
        self.index_intp = -1   # 插帧后的序号
        # 视频播放定时器
        self.intp_rate_timer_2_2 = 1
        self.timer_1 = QTimer()
        self.timer_1.timeout.connect(self.palyer_1)
        self.timer_2_1 = QTimer()
        self.timer_2_1.timeout.connect(self.player_2_1)
        self.timer_2_2 = QTimer()
        self.timer_2_2.timeout.connect(self.player_2_2)

        # 播放器的初始化
        self.player_init()
        # 播放器初始化
        self.is_switching = False
        self.is_pause = True
        # 菜单栏
        # 1：文件菜单栏
        # self.quit_sys.triggered.connect(qApp.quit)
        self.open_file_action.triggered.connect(self.open_file)
        self.open_mask_action.triggered.connect(self.open_mask)
        # 打开 待修复的图片后, 播放器初始化
        self.open_file_player_init_signal.connect(self.player_init)
        # 结果保存
        self.save_pictures_action.triggered.connect(self.result_save_pictures)
        self.save_video_action.triggered.connect(self.result_save_video)
        # 2: 窗口显示
        self.player_mode = 'all'
        self.functions_action.triggered.connect(self.show_sub_window_1)
        # 逐窗口视频播放
        # 按键
        self.start_stop_pushButton.clicked.connect(self.Btn_Start_Stop)
        self.quit_pushButton.clicked.connect(self.Btn_Quit)
        self.quit_signal.connect(self.show_single_frame)
        # create icons
        self.icon_start = QIcon('image/run.png')
        self.icon_pause = QIcon('image/pause.png')
        self.icon_stop = QIcon('image/stop.png')
        self.quit_pushButton.setIcon(self.icon_stop)
        # 上一帧 与 下一帧
        self.previous_frame_pushButton.clicked.connect(self.previous_frame)
        self.pervious_frame_signal.connect(self.show_single_frame)
        self.next_frame_pushButton.clicked.connect(self.next_frame)
        self.next_frame_signal.connect(self.show_single_frame)
        # 滑动按钮与显示之间的信号
        self.to_SN_Slider_signal.connect(self.set_SN_Slider)
        self.frame_SN_Slider.sliderReleased.connect(self.SN_Slider_to_others)
        self.SN_Slider_to_others_signal.connect(self.show_single_frame)
        # mask生成窗口（窗口2）的显示
        self.gen_masks_action.triggered.connect(self.show_sub_window_2)
        # 关闭窗口
        self.quit_sys_action.triggered.connect(self.closeEvent)

        # 快捷键
        # 键盘左键、右键 --> 上一帧，下一帧
        QShortcut(QKeySequence(Qt.Key_Left), self).activated.connect(self.previous_frame)
        QShortcut(QKeySequence(Qt.Key_Right), self).activated.connect(self.next_frame)
        # 播放
        QShortcut(QKeySequence(Qt.Key_Space), self).activated.connect(self.Btn_Start_Stop)
        # 退出
        QShortcut(QKeySequence(Qt.Key_Q), self).activated.connect(self.closeEvent)

        # 选择播放窗口
        self.player_flag = 0
        # self.player_all_flag = 0
        self.choose_player_pushButton.clicked.connect(self.choose_player)
        # 跟随按钮
        self.follow_flag = 0

    # 窗口选择按键：All(0) -->Left(1) -->Right(2) -->All(0)
    def choose_player(self):
        if self.player_flag < 2:
            self.player_flag = self.player_flag + 1
            if self.player_flag == 1:
                self.choose_player_pushButton.setText('Left')
                self.right_player_lineEdit.setText('')
            else:
                self.choose_player_pushButton.setText('Right')
                self.current_all_frames_lineEdit.setText('')
        else:
            self.player_flag = 0
            self.choose_player_pushButton.setText('All')
            if (self.inp_deno_deg_sr_inter_deold_stab_flags == [0, 0, 0, 0, 1, 0, 0]):
                pass
            else:
                self.right_player_lineEdit.setText('')
            # 保证同时播放时，初始序号为0
            self.index = 0
            self.index_2 = 0
            self.index_intp = 0

    # video to imgs
    def save_images(self, image, address, num):
        pic_address = os.path.join(address, "{:04}.jpg".format(num))
        cv2.imwrite(pic_address, image)

    def video_to_pic(self, video_path, save_path, frame_rate=1):
        # 读取视频文件
        videoCapture = cv2.VideoCapture(video_path)
        self.video_fps = int(videoCapture.get(5))
        self.video_frames_num = int(videoCapture.get(7))

        # 将视频拆为图片
        j = 0
        i = 0
        # 读帧
        success, frame = videoCapture.read()
        while success:
            i = i + 1
            # 每隔固定帧保存一张图片
            if i % frame_rate == 0:
                self.save_images(frame, save_path, j)
                j = j+1
            success, frame = videoCapture.read()

    # 在file.json中保存，打开的视频或者图片所在的文件夹，例如：/home/file/0001.jpg --> /home/file
    def open_file(self):
        self.inp_deno_deg_sr_inter_deold_stab_flags = [0, 0, 0, 0, 0, 0, 0]  # 保证播放器2不会输出上次产生的结果
        to_imgs_save_path = None
        config_file = os.path.join('config', 'file.json')

        with open(os.path.join('config', 'file.json'), 'r', encoding='utf-8') as f:
            config = json.loads(f.read())
        file_root = config['file_root']
        if not os.path.exists(file_root):
            file_root = os.getcwd()
        name, _ = QFileDialog.getOpenFileName(self, 'Video/image', file_root, "Pic File(*.mp4 *.mkv *.avi *.flv "
                                                                              "*.jpg *.png)")

        # 判断是视频还是视频帧，以及输入是否有问题
        suffix = name.split('.')[-1]
        if suffix in ['mp4', 'mkv', 'avi', 'flv']:
            # 主界面显示：正在处理输入数据
            self.current_all_frames_lineEdit.setText('请等待')
            # 将视频拆分成视频帧
            current_time = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())
            to_imgs_save_path = os.path.join('cache/video_to_imags', current_time)
            # 建立保存文件夹
            os.mkdir(to_imgs_save_path)
            self.video_to_pic(video_path=name, save_path=to_imgs_save_path)

            config['input_video_type'] = suffix
            config['input_video_num'] = self.video_frames_num
            config['input_video_fps'] = self.video_fps
            self.current_all_frames_lineEdit.setText('请使用')
        elif suffix in ['jpg', 'png']:
            # 方便保存时使用
            config['input_picture_type'] = suffix

        # 将输入视频帧的路径存放在 josn 中
        if to_imgs_save_path is not None:
            config['file_root'] = to_imgs_save_path
        else:
            config['file_root'] = os.path.dirname(name)
        config_json = json.dumps(config, ensure_ascii=False, indent=2)
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_json)
            print('2: config_json:', config_json)

        # 获取原图的尺寸,并存在json中
        file_name_list = os.listdir(config['file_root'])
        frame_dir = os.path.join(config['file_root'], file_name_list[0])
        image = cv2.imread(frame_dir)

        image_height, image_width = image.shape[0], image.shape[1]
        print('原图的高和宽为:', image_height, image_width)
        # 将 h, w 存入json
        config_file = os.path.join('config', 'file.json')
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        config["input_w"] = image_width
        config["input_h"] = image_height
        config_json = json.dumps(config, ensure_ascii=False, indent=2)
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_json)

        # 打开新的原图片后,信号发射,更新播放器 player
        self.open_file_player_init_signal.emit()
        # 从第1帧开始播放
        self.index = -1

    # 在file.json中保存，打开的视频或者图片所在的文件夹，例如：/home/file/0001.jpg --> /home/file
    def open_mask(self):
        config_file = os.path.join('config', 'file.json')
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        file_root = config['mask_root']
        if not os.path.exists(file_root):
            file_root = os.getcwd()
        name, _ = QFileDialog.getOpenFileName(self, 'Video/image', file_root, "Pic File(*.mp4 *.mkv *.avi *.flv "
                                                                              "*.jpg *.png)")
        print('3: name:', name)
        if name:
            config['mask_root'] = os.path.dirname(name)
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)
                print('4: config_json:', config_json)

    # 显示 修复的子窗口
    def show_sub_window_1(self):
        sub_window_1 = SubWindow_1(self)
        sub_window_1.show()
        # 下面的显示方法也可以。
        # sub_window_1 = SubWindow_1()
        # sub_window_1.exec()

    def show_sub_window_2(self):
        sub_window_2 = SubWindow_2()
        sub_window_2.show()

    def Btn_Start_Stop(self):
        if self.is_pause or self.is_switching:
            self.start_stop_pushButton.setIcon(self.icon_pause)
            self.is_pause = False
            self.Btn_Start()
        elif (not self.is_pause) and (not self.is_switching):
            self.start_stop_pushButton.setIcon(self.icon_start)
            self.is_pause = True
            self.Btn_Stop()

    # 定时器开启，每隔一段时间，读取一帧
    def Btn_Start(self):
        if (self.player_flag == 0) or (self.player_flag == 2):
            if sum(myWin.inp_deno_deg_sr_inter_deold_stab_flags) > 0:
                if myWin.inp_deno_deg_sr_inter_deold_stab_flags == [0, 0, 0, 0, 1, 0, 0]:
                    self.timer_2_2.start(40 // self.intp_rate_timer_2_2)
                else:
                    self.timer_2_1.start(40)
            if not(self.player_flag == 2):
                self.timer_1.start(40)
        elif self.player_flag == 1:
            self.timer_1.start(40)

    def Btn_Stop(self):
        if self.timer_1.isActive():
            self.timer_1.stop()
        if self.timer_2_1.isActive():
            self.timer_2_1.stop()
        if self.timer_2_2.isActive():
            self.timer_2_2.stop()

    def Btn_Quit(self):
        self.start_stop_pushButton.setIcon(self.icon_start)
        self.is_pause = True
        if self.timer_1.isActive():
            self.timer_1.stop()
        if self.timer_2_1.isActive():
            self.timer_2_1.stop()
        if self.timer_2_2.isActive():
            self.timer_2_2.stop()

        if (self.player_flag == 0) or (self.player_flag == 2):
            if self.player_flag == 0:
                self.index = 0
                self.quit_signal.emit('player_1')
            if sum(self.inp_deno_deg_sr_inter_deold_stab_flags) > 0:
                if self.inp_deno_deg_sr_inter_deold_stab_flags == [0, 0, 0, 0, 1, 0, 0]:
                    self.index_intp = 0
                    self.quit_signal.emit('player_2_2')
                else:
                    self.index_2 = 0
                    self.quit_signal.emit('player_2_1')
        elif self.player_flag == 1:
            self.index = 0
            self.quit_signal.emit('player_1')

    def player_init(self):
        try:
            # 取出保存的 frames 和 masks 的地址
            with open(os.path.join('config', 'file.json'), 'r', encoding='utf-8') as f:
                file = json.loads(f.read())
            file_root = file['file_root']

            self.frame_path = file_root
            self.frame_name_list = os.listdir(self.frame_path)
            # 确保视频帧顺序读取
            self.frame_name_list.sort()
            print(self.frame_name_list)
            self.frame_len = len(self.frame_name_list)

            # self.all_frames_lineEdit.setText(str(self.frame_len))
            self.frame_SN_Slider.setMinimum(1)
            self.frame_SN_Slider.setMaximum(self.frame_len)
        except Exception as e:
            print(e)

    # 窗口1播放，播放原始视频帧
    def palyer_1(self):
        if self.index == self.frame_len - 1:
            self.index = 0
        else:
            self.index = self.index + 1
        # 设置当前帧的显示
        self.set_text(serial=1)
        # 滑动栏触发
        if self.player_flag == 0:
            if self.inp_deno_deg_sr_inter_deold_stab_flags == [0, 0, 0, 0, 1, 0, 0]:
                self.frame_SN_Slider.setEnabled(False)  # all模式且插帧模式下，滑动条固定不动。
            else:
                self.to_SN_Slider_signal.emit('ori')
        elif self.player_flag == 1:
            self.to_SN_Slider_signal.emit('ori')

        frame_dir = os.path.join(self.frame_path, self.frame_name_list[self.index])
        image = cv2.imread(frame_dir)
        image_height, image_width = image.shape[0], image.shape[1]
        long_w_h = image_width - image_height
        if long_w_h >= 0:
            #  w > h，加宽 h
             image_show = cv2.copyMakeBorder(image, int(long_w_h/2), int(long_w_h/2), 0, 0,
                                             cv2.BORDER_CONSTANT, value=[0, 0, 0])
        else:
            image_show = cv2.copyMakeBorder(image, 0, 0, int(long_w_h / 2), int(long_w_h / 2),
                                            cv2.BORDER_CONSTANT, value=[0, 0, 0])
        try:
            if len(image_show.shape) == 3:
                image_show = cv2.cvtColor(image_show, cv2.COLOR_BGR2RGB)
                # video_img = QImage(image_show.data, image_show.shape[1], image_show.shape[0], QImage.Format_RGB888)
                video_img = QImage(image_show.data, image_show.shape[1], image_show.shape[0], image_show.shape[1]*3, QImage.Format_RGB888)
            elif len(image_show.shape) == 1:
                video_img = QImage(image_show.data, image_show.shape[1], image_show.shape[0], QImage.Format_Indexed8)
            else:
                video_img = QImage(image_show.data, image_show.shape[1], image_show.shape[0], QImage.Format_RGB888)

            self.video_player_label_1.setPixmap(QPixmap(video_img))
            self.video_player_label_1.setScaledContents(True)  # 自适应窗口
        except Exception as e:
            print(e)

    # 窗口2播放，播放原始修复后的视频帧，除了插帧后的视频帧
    def player_2_1(self):
        if self.index_2 == self.frame_len - 1:
            self.index_2 = 0
        else:
            self.index_2 = self.index_2 + 1
        # 保证 All 模式下，左侧的序号不动
        if self.player_flag == 2:
            self.set_text(serial=2_1)
            self.to_SN_Slider_signal.emit('2_1')  # 滑动栏触发

        completed_frames_dir = os.path.join(self.completed_frames_path, self.completed_frames_name_list[self.index_2])
        completed_image = cv2.imread(completed_frames_dir)
        image_height, image_width = completed_image.shape[0], completed_image.shape[1]
        long_w_h = image_width - image_height
        if long_w_h >= 0:
            completed_image_show = cv2.copyMakeBorder(completed_image, int(long_w_h / 2), int(long_w_h / 2), 0, 0,
                                                      cv2.BORDER_CONSTANT, value=[0, 0, 0])
        else:
            completed_image_show = cv2.copyMakeBorder(completed_image, 0, 0, int(long_w_h / 2), int(long_w_h / 2),
                                                      cv2.BORDER_CONSTANT, value=[0, 0, 0])
        # 修复的视频帧 展示
        try:
            if len(completed_image_show.shape) == 3:
                completed_image_show = cv2.cvtColor(completed_image_show, cv2.COLOR_BGR2RGB)
                video_completed_img = QImage(completed_image_show.data, completed_image_show.shape[1],
                                             completed_image_show.shape[0],
                                             QImage.Format_RGB888)
            elif len(completed_image_show.shape) == 1:
                video_completed_img = QImage(completed_image_show.data, completed_image_show.shape[1],
                                             completed_image_show.shape[0],
                                             QImage.Format_Indexed8)
            else:
                video_completed_img = QImage(completed_image_show.data, completed_image_show.shape[1],
                                             completed_image_show.shape[0],
                                             QImage.Format_RGB888)

            self.video_player_label_2.setPixmap(QPixmap(video_completed_img))
            self.video_player_label_2.setScaledContents(True)  # 自适应窗口
        except Exception as e:
            print(e)

    # 窗口2播放，播放插帧后的视频帧
    def player_2_2(self):
        if self.index_intp == ((self.frame_len - 1) * self.intp_rate_timer_2_2 + 1) - 1:
            self.index_intp = 0
        else:
            self.index_intp = self.index_intp + 1

        self.set_text(serial=2_2)
        # if self.player_flag == 2:
        #     self.to_SN_Slider_signal.emit()  # 滑动栏触发

        completed_frames_dir = os.path.join(self.completed_frames_path,
                                            self.completed_frames_name_list[self.index_intp])
        completed_image = cv2.imread(completed_frames_dir)
        image_height, image_width = completed_image.shape[0], completed_image.shape[1]
        long_w_h = image_width - image_height
        if long_w_h >= 0:
            completed_image_show = cv2.copyMakeBorder(completed_image, int(long_w_h / 2), int(long_w_h / 2), 0, 0,
                                                      cv2.BORDER_CONSTANT, value=[0, 0, 0])
        else:
            completed_image_show = cv2.copyMakeBorder(completed_image, 0, 0, int(long_w_h / 2), int(long_w_h / 2),
                                                      cv2.BORDER_CONSTANT, value=[0, 0, 0])
        # 修复的视频帧 展示
        try:
            if len(completed_image_show.shape) == 3:
                completed_image_show = cv2.cvtColor(completed_image_show, cv2.COLOR_BGR2RGB)
                video_completed_img = QImage(completed_image_show.data, completed_image_show.shape[1],
                                             completed_image_show.shape[0],
                                             QImage.Format_RGB888)
            elif len(completed_image_show.shape) == 1:
                video_completed_img = QImage(completed_image_show.data, completed_image_show.shape[1],
                                             completed_image_show.shape[0],
                                             QImage.Format_Indexed8)
            else:
                video_completed_img = QImage(completed_image_show.data, completed_image_show.shape[1],
                                             completed_image_show.shape[0],
                                             QImage.Format_RGB888)

            self.video_player_label_2.setPixmap(QPixmap(video_completed_img))
            self.video_player_label_2.setScaledContents(True)  # 自适应窗口
        except Exception as e:
            print(e)

    # 显示单独的一帧
    def show_single_frame(self, mode='player_1'):
        # model:player_1, player_2_1, player_2_1
        try:
            if mode == 'player_1':
                self.set_text(serial=1)
                frame_dir = os.path.join(self.frame_path, self.frame_name_list[self.index])
            elif mode == 'player_2_1':
                if self.player_flag == 2:
                    self.set_text(serial=2_1)
                frame_dir = os.path.join(self.completed_frames_path, self.completed_frames_name_list[self.index_2])
            elif mode == 'player_2_2':
                self.set_text(serial=2_2)
                frame_dir = os.path.join(self.completed_frames_path, self.completed_frames_name_list[self.index_intp])

            # 视频帧的读取
            image = cv2.imread(frame_dir)
            image_height, image_width = image.shape[0], image.shape[1]
            long_w_h = image_width - image_height
            if long_w_h >= 0:
                #  w > h，加宽 h
                image_show = cv2.copyMakeBorder(image, int(long_w_h / 2), int(long_w_h / 2), 0, 0,
                                                cv2.BORDER_CONSTANT, value=[0, 0, 0])
            else:
                image_show = cv2.copyMakeBorder(image, 0, 0, int(long_w_h / 2), int(long_w_h / 2),
                                                cv2.BORDER_CONSTANT, value=[0, 0, 0])
            if True:
                if len(image_show.shape) == 3:
                    image_show = cv2.cvtColor(image_show, cv2.COLOR_BGR2RGB)
                    video_img = QImage(image_show.data, image_show.shape[1], image_show.shape[0], QImage.Format_RGB888)
                elif len(image_show.shape) == 1:
                    video_img = QImage(image_show.data, image_show.shape[1], image_show.shape[0], QImage.Format_Indexed8)
                else:
                    video_img = QImage(image_show.data, image_show.shape[1], image_show.shape[0], QImage.Format_RGB888)
            # 视频帧的展示
            if mode == 'player_1':
                self.video_player_label_1.setPixmap(QPixmap(video_img))
                self.video_player_label_1.setScaledContents(True)  # 自适应窗口
            else:
                self.video_player_label_2.setPixmap(QPixmap(video_img))
                self.video_player_label_2.setScaledContents(True)  # 自适应窗口
        except Exception as e:
            print(e)

    def previous_frame(self):
        if self.player_flag == 0:   # 双窗口播放模式
            if self.inp_deno_deg_sr_inter_deold_stab_flags == [0, 0, 0, 0, 1, 0, 0]:
                pass    # all模式下，插帧时，上一帧和下一帧按键无反应
            else:
                if self.index >= 1:
                    self.index = self.index - 1
                if sum(self.inp_deno_deg_sr_inter_deold_stab_flags) > 0:
                    if self.index_2 >= 1:
                        self.index_2 = self.index_2 - 1
                    self.pervious_frame_signal.emit('player_2_1')
                self.pervious_frame_signal.emit('player_1')
        elif self.player_flag == 1:     # 做窗口播放
            if self.index >= 1:
                self.index = self.index - 1
            self.pervious_frame_signal.emit('player_1')
        elif self.player_flag == 2:     # 右窗口播放
            if sum(self.inp_deno_deg_sr_inter_deold_stab_flags) > 0:
                if self.inp_deno_deg_sr_inter_deold_stab_flags == [0, 0, 0, 0, 1, 0, 0]:
                    if self.index_intp >= 1:
                        self.index_intp = self.index_intp - 1
                        self.pervious_frame_signal.emit('player_2_2')
                else:
                    if self.index_2 >= 1:
                        self.index_2 = self.index_2 - 1
                    self.pervious_frame_signal.emit('player_2_1')
            else:
                pass

    def next_frame(self):
        if self.player_flag == 0:   # 双窗口播放模式
            if self.inp_deno_deg_sr_inter_deold_stab_flags == [0, 0, 0, 0, 1, 0, 0]:
                pass    # all模式下，插帧时，上一帧和下一帧按键无反应
            else:
                if self.index < self.frame_len - 1:
                    self.index = self.index + 1
                if sum(self.inp_deno_deg_sr_inter_deold_stab_flags) > 0:
                    if self.index_2 < self.frame_len - 1:
                        self.index_2 = self.index_2 + 1
                    self.pervious_frame_signal.emit('player_2_1')
                self.pervious_frame_signal.emit('player_1')
        elif self.player_flag == 1:     # 左窗口播放
            if self.index < self.frame_len - 1:
                self.index = self.index + 1
            self.pervious_frame_signal.emit('player_1')
        elif self.player_flag == 2:     # 右窗口播放
            if sum(self.inp_deno_deg_sr_inter_deold_stab_flags) > 0:
                if self.inp_deno_deg_sr_inter_deold_stab_flags == [0, 0, 0, 0, 1, 0, 0]:
                    if self.index_intp < ((self.frame_len - 1) * self.intp_rate_timer_2_2 + 1) - 1:
                        self.index_intp = self.index_intp + 1
                        self.pervious_frame_signal.emit('player_2_2')
                else:
                    if self.index < self.frame_len - 1:
                        self.index_2 = self.index_2 + 1
                    self.pervious_frame_signal.emit('player_2_1')
            else:
                pass

    def set_text(self, serial):
        if serial == 1:
            text = '%s / %s' % (str(self.index + 1), str(self.frame_len))
            self.current_all_frames_lineEdit.setText(text)
        elif serial == 2_1:
            text = '%s / %s' % (str(self.index_2 + 1), str(self.frame_len))
            self.right_player_lineEdit.setText(text)
        elif serial == 2_2:
            text = '%s / %s' % (str(self.index_intp + 1), str(((self.frame_len - 1) * self.intp_rate_timer_2_2 + 1)))
            self.right_player_lineEdit.setText(text)
        elif serial == 10:
            self.current_all_frames_lineEdit.setText('')
        elif serial == 20:
            self.right_player_lineEdit.setText('')

    def set_SN_Slider(self, mode='ori'):
        if mode == 'ori':
            self.frame_SN_Slider.setValue(self.index+1)
        elif mode == '2_1':
            self.frame_SN_Slider.setValue(self.index_2 + 1)

    # 拖动结束后，才会视频和滑动栏的显示才会有变化。
    def SN_Slider_to_others(self):
        self.index = self.frame_SN_Slider.value() - 1
        # 触发 Open_Single_Frame
        self.SN_Slider_to_others_signal.emit()  # todo 2023.5.10

    # 将取出存放在 config/file.json中的：修复后视频帧存放的地址
    # 为 player2 的播放做准备
    def completed_frames_init(self):
        with open(os.path.join('config', 'file.json'), 'r', encoding='utf-8') as f:
            file = json.loads(f.read())
        output_root = file['completed_frames_root']
        self.completed_frames_path = output_root
        self.completed_frames_name_list = os.listdir(self.completed_frames_path)
        self.completed_frames_name_list.sort()
        self.completed_frames_len = len(self.completed_frames_name_list)
        self.index = -1     # 修复后，播放器从第一帧开始播放

    # 保存为图片，将'completed_frames_root'中的图片，直接复制到用户选择的位置
    def result_save_pictures(self):
        obj_path = str(QFileDialog.getExistingDirectory(self, "请选择保存路径"))
        with open(os.path.join('config', 'file.json'), 'r', encoding='utf-8') as f:
            file = json.loads(f.read())
        scr_path = file['completed_frames_root']

        filelist = os.listdir(scr_path)
        for item in filelist:
            src = os.path.join(os.path.abspath(scr_path), item)
            dst = os.path.join(os.path.abspath(obj_path), item)
            shutil.copy(src, dst)  # 将src复制到dst

        self.current_all_frames_lineEdit.setText('保存完成')

    # 保存为视频，将'completed_frames_root'中的图片先合成视频，然后再存放到相应位置。统一保存为 .mp4 格式。
    def result_save_video(self):
        # 1:设置源路径与保存路径
        save_path = str(QFileDialog.getExistingDirectory(self, "请选择保存路径"))
        save_name = 'result.mp4'
        save_path = os.path.join(save_path, save_name)

        with open(os.path.join('config', 'file.json'), 'r', encoding='utf-8') as f:
            file = json.loads(f.read())
        src_path = file['completed_frames_root']

        # todo 保存时的帧率。插帧的帧率需要注意，其他的功能，正常对待。
        if self.inp_deno_deg_sr_inter_deold_stab_flags == [0, 0, 0, 0, 1, 0, 0]:
            output_video_fps = file['input_video_fps'] * file['interpolation_rate']
        else:
            output_video_fps = file['input_video_fps']

        # todo 输出尺寸的设置。超分需要单独注意，其他功能的输出尺寸应该没有变化。
        if self.inp_deno_deg_sr_inter_deold_stab_flags == [0, 0, 0, 1, 0, 0, 0]:
            pass
        else:
            output_w = file['input_w']
            output_h = file['input_h']

        # 2.每张图像大小
        size = (output_w, output_h)
        # print("每张图片的大小为({},{})".format(size[0], size[1]))

        # 3.获取文件夹中所有图片的名称，个数
        all_filename = os.listdir(src_path)
        all_filename.sort()
        number = len(all_filename)

        # 4.设置视频写入器，保存为 mp4格式
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4格式

        # 完成写入对象的创建，第一个参数是合成之后的视频的名称，第二个参数是可以使用的编码器，
        # 第三个参数是帧率即每秒钟展示多少张图片，第四个参数是图片大小信息
        videowrite = cv2.VideoWriter(save_path, fourcc, output_video_fps, size)  # size是图片尺寸

        # 5.临时存放图片的数组
        # 读取文件夹中的所有的图片
        img_array = []
        for j in range(number):
            filename = os.path.join(src_path, all_filename[j])
            img = cv2.imread(filename)
            if img is None:
                print(filename + " is error!")
                continue
            img_array.append(img)

        # 7.合成视频
        for i in range(0, number):
            img_array[i] = cv2.resize(img_array[i], size)
            videowrite.write(img_array[i])

        self.current_all_frames_lineEdit.setText('保存完成')

    # 关闭主窗口的提示，以及所需要做的相关操作
    def delete_files(self, dir_path):
        # os.walk会得到dir_path下各个后代文件夹和其中的文件的三元组列表，顺序自内而外排列，
        for root, dirs, files in os.walk(dir_path, topdown=False):
            # print(root)  # 各级文件夹绝对路径
            # print(dirs)  # root下一级文件夹名称列表，如 ['文件夹1','文件夹2']
            # print(files)  # root下文件名列表，如 ['文件1','文件2']
            # 先删除所有文件，再删除所有空文件
            # 第一步：删除文件
            for name in files:
                os.remove(os.path.join(root, name))  # 删除文件
            # 第二步：删除空文件夹
            for name in dirs:
                os.rmdir(os.path.join(root, name))  # 删除一个空目录

    def closeEvent(self, event=False):
        reply = QMessageBox.question(self, '提示',
                                     "是否要关闭所有窗口?",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            # 关闭前先执行相关操作
            # 清理相关缓存文件
            delete_path_1 = 'cache/completed_frames'
            delete_path_2 = 'cache/video_to_imags'
            delete_path_3 = 'cache/inp_optimization_frames'
            self.delete_files(delete_path_1)
            self.delete_files(delete_path_2)
            self.delete_files(delete_path_3)
            # 退出程序
            event.accept()
            sys.exit(0)
        else:
            # 其他的方式引发的退出，则 event 为 False
            if event == False:
                pass
            else:
                event.ignore()  # 右上角点击，则event则存在


# 工具窗口
class SubWindow_1(QDialog, Ui_Dialog_tool):

    inp_optimized_frames_change_signal = pyqtSignal()

    def __init__(self, parent=None):
        super(SubWindow_1, self).__init__(parent)
        self.setupUi(self)
        # 测试使用
        # self.i = 0
        # 修复提示栏的初始状态为：''
        # self.sub1_console_push_text('')

        # 各种功能的线程，进行实例化
        self.inp_thread = inp_Thread()
        self.sr_thread = sr_Thread()
        self.denoising_thread = denoising_Thread()
        self.intp_thread = intp_Thread()
        self.res_thread = restore_Thread()
        self.color_thread = colorization_Thread()
        init_config()

        # 跟随按钮
        self.sub1_follow_pushButton.clicked.connect(self.follow_state)

        # 修复的初始模型名为: DSTT
        self.inp_model_name = 'DSTT'
        # 与修复相关的信号，去绑定槽函数
        self.sub1_comboBox.currentIndexChanged.connect(self.inp_model_change)
        self.sub1_open_mask_pushButton.clicked.connect(self.open_mask)
        self.sub1_mask_gen_pushButton.clicked.connect(self.show_sub_window_2)
        # 修复中的优化
        self.sub1_inp_opti_flag = 0
        self.sub1_remove_opt_frames_itextEdit.setDisabled(True)
        self.sub1_opti_add_pushButton.setDisabled(True)
        self.sub1_opti_dec_pushButton.setDisabled(True)
        self.sub1_opti_clear_pushButton.setDisabled(True)
        self.sub1_inp_opt_choose_radioButton.toggled.connect(self.inp_opti_enable_display)

        self.inp_optimized_frames = []
        self.sub1_opti_add_pushButton.clicked.connect(self.add_inp_optimized_frames)
        self.sub1_opti_dec_pushButton.clicked.connect(self.dec_inp_optimized_frames)
        self.sub1_opti_clear_pushButton.clicked.connect(self.clear_inp_optimized_frames)
        self.inp_optimized_frames_change_signal.connect(self.show_sub1_inp_optimized_frames)
        # 优化相关的快捷键
        # QShortcut(QKeySequence(Qt.Key_Left), self).activated.connect(myWin.previous_frame)
        # QShortcut(QKeySequence(Qt.Key_Left), self).activated.connect(myWin.next_frame)
        # QShortcut(QKeySequence(Qt.Key_C), self).activated.connect(self.clear_inp_optimized_frames)


        # 超分的相关设置
        # 超分的初始模型名为：BasicVsr
        self.sr_model_name = 'BasicVsr'
        self.sub1_sr_comboBox.currentIndexChanged.connect(self.sr_model_change)

        # 去噪的相关设置
        # todo 4.11 修改 by Chen
        self.denoising_model_name = 'fastdvdnet'
        # self.sub1_deno_radioButton.toggled.connect(self.deno_model)
        self.deno_model()
        # 去噪强度滑动条相关设置
        self.denoise_Slider.setMaximum(50)
        self.denoise_Slider.setMinimum(1)
        self.denoise_Slider.setTickPosition(QSlider.TicksBelow)
        self.denoise_Slider.valueChanged.connect(self.set_deno_strength)
        self.max_num_fr_per_seq_spinBox.setValue(90)
        self.max_num_fr_per_seq_spinBox.setSingleStep(5)
        self.max_num_fr_per_seq_spinBox.setMaximum(500)
        self.max_num_fr_per_seq_spinBox.valueChanged.connect(self.set_max_num_fr_per_seq)

        # 插帧相关
        self.intp_model_name = 'RIFE'
        self.sub1_intp_comboBox.currentIndexChanged.connect(self.set_intp_rate)
        self.sub1_intp_comboBox.currentIndexChanged.connect(self.display_fps)
        self.intp_is4k_checkBox.toggled.connect(self.is4k)
        self.save_video_Button.clicked.connect(self.save_video)

        # 去污相关
        self.res_model_name = 'remaster'
        self.mindim_Slider.setMaximum(500)
        self.mindim_Slider.setMinimum(200)
        self.mindim_Slider.setValue(320)
        self.mindim_Slider.valueChanged.connect(self.set_mindim)

        # 上色相关
        self.color_model_name = 'SVCNet'

        self.crop_size_w.setMinimum(256)
        self.crop_size_w.setMaximum(832)
        self.crop_size_w.setValue(448)
        self.crop_size_w_spinBox_2.setRange(256, 832)
        self.crop_size_w_spinBox_2.setValue(448)
        self.crop_size_w.valueChanged.connect(self.set_crop_size_w_spin)
        self.crop_size_w_spinBox_2.valueChanged.connect(self.set_crop_size_w_slider)

        self.crop_size_h.setMinimum(128)
        self.crop_size_h.setMaximum(448)
        self.crop_size_h.setValue(256)
        self.crop_size_h_spinBox.setRange(128, 448)
        self.crop_size_h_spinBox.setValue(256)
        self.crop_size_h.valueChanged.connect(self.set_crop_size_h_spin)
        self.crop_size_h_spinBox.valueChanged.connect(self.set_crop_size_h_slider)

        self.color_frame_button.clicked.connect(self.ref_frame)
        myWin.color_completed_signal.connect(self.open_colored_image)

        self.color_image_right_button.clicked.connect(self.color_image_right)
        self.color_image_left_button.clicked.connect(self.color_image_left)
        self.color_image_right_button.clicked.connect(self.open_colored_image)
        self.color_image_left_button.clicked.connect(self.open_colored_image)
        myWin.color_error_signal.connect(self.color_error)
        # 线程完成后，会发射这个信号，表示已经完成了，从而窗口1的右下角的控制台会显示“已完成”。
        myWin.function_completed_signal.connect(self.function_completed)
        # 点击这个按钮后，相应的功能开始执行
        self.sub1_start_pushButton.clicked.connect(self.start_pushbutton_run)

    # 跟随按钮对应的槽函数
    def follow_state(self):
        if os.listdir(os.path.join('cache', 'completed_frames')):
            myWin.follow_flag = 1
            self.sub1_console_push_text('跟随状态')
        else:
            self.sub1_console_push_text('请先执行相应的功能，再使用跟随操作')

    # 当前功能的 flag=1, 其余为0
    def choose_function(self):
        if self.sub1_tools_tabWidget.currentIndex() == 0:
            myWin.inp_deno_deg_sr_inter_deold_stab_flags = [0, 1, 0, 0, 0, 0, 0]
        elif self.sub1_tools_tabWidget.currentIndex() == 1:
            myWin.inp_deno_deg_sr_inter_deold_stab_flags = [0, 0, 1, 0, 0, 0, 0]
        elif self.sub1_tools_tabWidget.currentIndex() == 2:
            myWin.inp_deno_deg_sr_inter_deold_stab_flags = [0, 0, 0, 0, 0, 0, 1]
        elif self.sub1_tools_tabWidget.currentIndex() == 3:
            myWin.inp_deno_deg_sr_inter_deold_stab_flags = [1, 0, 0, 0, 0, 0, 0]
        elif self.sub1_tools_tabWidget.currentIndex() == 4:
            myWin.inp_deno_deg_sr_inter_deold_stab_flags = [0, 0, 0, 1, 0, 0, 0]
        elif self.sub1_tools_tabWidget.currentIndex() == 5:
            myWin.inp_deno_deg_sr_inter_deold_stab_flags = [0, 0, 0, 0, 1, 0, 0]
        elif self.sub1_tools_tabWidget.currentIndex() == 6:
            myWin.inp_deno_deg_sr_inter_deold_stab_flags = [0, 0, 0, 0, 0, 1, 0]

    # 窗口 1 右下角控制台的显示
    def sub1_console_push_text(self, text):
        text = '--> %s' %(text)
        self.sub1_console_plainTextEdit.appendPlainText(text)
        self.sub1_console_plainTextEdit.moveCursor(QTextCursor.End)

    # 窗口 1 中的 开始按钮
    def start_pushbutton_run(self):
        # flags
        self.choose_function()
        self.sub1_console_push_text('正在进行中...')
        # 看一看什么功能被选中了，执行相应的线程。
        #                                            去噪，去污损，稳相，目标移除，超分，插帧，上色
        #  self.sub1_tools_tabWidget.currentIndex()：  0    1     2      3      4    5    6

        if self.sub1_tools_tabWidget.currentIndex() == 3:
            # 将选择的修复模型名赋值给线程里的变量
            self.inp_thread.inp_model_name_input = self.inp_model_name
            # 判断是否为优化模式
            if self.sub1_inp_opti_flag == 1:
                # 确保选择了待优化的视频帧
                if self.inp_optimized_frames != []:
                    self.inp_thread.inp_optimization_flag = self.sub1_inp_opti_flag
                    self.inp_thread.inp_optimization_index = self.inp_optimized_frames
                    self.completed_frames_save_root(mode='inp_opti')
                    # 线程运行
                    if not self.inp_thread.isRunning():
                        self.inp_thread.start()
                else:
                    self.sub1_console_push_text('请增加待优化的视频帧')
            else:
                self.completed_frames_save_root()
                # 线程运行
                if not self.inp_thread.isRunning():
                    self.inp_thread.start()

        elif self.sub1_tools_tabWidget.currentIndex() == 4:
            self.completed_frames_save_root()
            self.sr_thread.sr_model_name_input = self.sr_model_name
            if not self.sr_thread.isRunning():
                self.sr_thread.start()

        elif self.sub1_tools_tabWidget.currentIndex() == 0:
            self.completed_frames_save_root()
            self.denoising_thread.denoising_model_name_input = self.denoising_model_name
            if not self.denoising_thread.isRunning():
                self.denoising_thread.start()

        elif self.sub1_tools_tabWidget.currentIndex() == 5:
            self.completed_frames_save_root()
            if not self.intp_thread.isRunning():
                self.intp_thread.start()

        elif self.sub1_tools_tabWidget.currentIndex() == 1:
            self.completed_frames_save_root()
            if not self.res_thread.isRunning():
                self.res_thread.start()

        elif self.sub1_tools_tabWidget.currentIndex() == 6:
            self.completed_frames_save_root()
            if not self.color_thread.isRunning():
                self.color_thread.start()

    # 与修复相关的函数
    # 根据sub1_combox的 序号 来确定 model_name
    def inp_model_change(self):
        if self.sub1_comboBox.currentIndex() == 0:
            self.inp_model_name = 'DSTT'
        elif self.sub1_comboBox.currentIndex() == 1:
            self.inp_model_name = 'FuseFormer'
        elif self.sub1_comboBox.currentIndex() == 2:
            self.inp_model_name = 'FGT'

        inp_model_name_text = '模式：%s' % (self.sub1_comboBox.currentText())
        self.sub1_console_push_text(inp_model_name_text)

    # 打开mask
    def open_mask(self):
        config_file = os.path.join('config', 'file.json')
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        file_root = config['mask_root']
        if not os.path.exists(file_root):
            file_root = os.getcwd()
        name, _ = QFileDialog.getOpenFileName(self, 'Video/image', file_root, "Pic File(*.mp4 *.mkv *.avi *.flv "
                                                                              "*.jpg *.png)")
        print('3: name:', name)
        if name:
            config['mask_root'] = os.path.dirname(name)
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)
                print('4: config_json:', config_json)

    # mask生成
    def show_sub_window_2(self):
        sub_window_2 = SubWindow_2()
        sub_window_2.show()

    # 增加待优化的帧
    def add_inp_optimized_frames(self):
        if myWin.index not in self.inp_optimized_frames:
            self.inp_optimized_frames.append(myWin.index)
        self.inp_optimized_frames.sort()
        self.inp_optimized_frames_change_signal.emit()
        print("优化帧为：", self.inp_optimized_frames)

    def dec_inp_optimized_frames(self):
        if myWin.index in self.inp_optimized_frames:
            self.inp_optimized_frames.remove(myWin.index)
        self.inp_optimized_frames.sort()
        self.inp_optimized_frames_change_signal.emit()
        print("优化帧为：", self.inp_optimized_frames)

    def clear_inp_optimized_frames(self):
        self.inp_optimized_frames = []
        self.inp_optimized_frames_change_signal.emit()

    def show_sub1_inp_optimized_frames(self):
        text = [i+1 for i in self.inp_optimized_frames]
        text = str(text).strip('[').strip(']')

        self.sub1_remove_opt_frames_itextEdit.setText(text)

    def inp_opti_enable_display(self):
        if self.sub1_inp_opt_choose_radioButton.isChecked():
            self.sub1_inp_opti_flag = 1
            self.sub1_remove_opt_frames_itextEdit.setDisabled(False)
            self.sub1_opti_add_pushButton.setDisabled(False)
            self.sub1_opti_dec_pushButton.setDisabled(False)
            self.sub1_opti_clear_pushButton.setDisabled(False)
        else:
            self.sub1_inp_opti_flag = 0
            self.sub1_remove_opt_frames_itextEdit.setDisabled(True)
            self.sub1_opti_add_pushButton.setDisabled(True)
            self.sub1_opti_dec_pushButton.setDisabled(True)
            self.sub1_opti_clear_pushButton.setDisabled(True)

    #超分相关的函数
    def sr_model_change(self):
        if self.sub1_sr_comboBox.currentIndex() == 0:
            self.inp_model_name = 'BasicVsr'
        elif self.sub1_sr_comboBox.currentIndex() == 1:
            self.inp_model_name = 'FuseFormer'
        elif self.sub1_sr_comboBox.currentIndex() == 2:
            self.inp_model_name = 'FGT'

        sr_model_name_text = '模式：%s' %(self.sub1_sr_comboBox.currentText())
        self.sub1_console_push_text(sr_model_name_text)

    # 去噪相关的函数
    def deno_model(self):
        self.denoising_model_name = 'fastdvdnet'
        deno_model_name_text = '模式：%s' %(self.denoising_model_name)
        # self.sub1_console_push_text(deno_model_name_text)

    def set_deno_strength(self):

        write_config("deno_strength", self.denoise_Slider.value())

    def set_max_num_fr_per_seq(self):
        write_config("max_num_fr_per_seq", self.max_num_fr_per_seq_spinBox.value())

    # 插帧相关函数
    def set_intp_rate(self):
        if self.sub1_intp_comboBox.currentIndex() == 0:
            self.intp_rate = 2
        if self.sub1_intp_comboBox.currentIndex() == 1:
            self.intp_rate = 4
        if self.sub1_intp_comboBox.currentIndex() == 2:
            self.intp_rate = 8
        write_config("interpolation_rate", self.intp_rate)
        intp_model_name_text = '模式：%s倍插帧' % (self.intp_rate)
        self.sub1_console_push_text(intp_model_name_text)

    def is4k(self):
        if self.intp_is4k_checkBox.isChecked():
            write_config("UHD", True)
        else:
            write_config("UHD", False)

    def display_fps(self):
        rate = read_config("interpolation_rate")
        self.fps_label.setText(str(25 * rate))

    def save_video(self):
        save_path, _ =QFileDialog.getSaveFileName(self, caption="保存视频",  filter="保存.mp4文件(*.mp4)", directory='video.mp4')
        img2video(save_path)


    # 去污相关的函数
    def set_mindim(self):
        write_config("mindim", self.mindim_Slider.value())
        self.sub1_console_push_text('模式：去污')

    # 上色相关的函数
    def set_lambda_spin(self):
        self.spinBox.setValue(self.lambda_Slider.value())
        write_config("lambda_value", self.lambda_Slider.value())

    def set_lambda_slider(self):
        self.lambda_Slider.setValue(self.spinBox.value())
        write_config("lambda_value", self.spinBox.value())

    def set_sigma_spin(self):
        self.sigma_spinBox.setValue(self.sigma_Slider.value())
        write_config("sigma_color", self.sigma_Slider.value())

    def set_sigma_slider(self):
        self.sigma_Slider.setValue(self.sigma_spinBox.value())
        write_config("sigma_color", self.sigma_Slider.value())

    def set_iter_frames_spin(self):
        self.iter_frames_spinBox.setValue(self.iter_frames_Slider.value())
        write_config("iter_frames", self.iter_frames_Slider.value())

    def set_iter_frames_slider(self):
        self.iter_frames_Slider.setValue(self.iter_frames_spinBox.value())
        write_config("iter_frames", self.iter_frames_spinBox.value())

    def set_crop_size_h_spin(self):
        self.crop_size_h_spinBox.setValue(self.crop_size_h.value())
        write_config("crop_size_h", self.crop_size_h.value())

    def set_crop_size_h_slider(self):
        self.crop_size_h.setValue(self.crop_size_h_spinBox.value())
        write_config("crop_size_h", self.crop_size_h_spinBox.value())

    def set_crop_size_w_spin(self):
        self.crop_size_w_spinBox_2.setValue(self.crop_size_w.value())
        write_config("crop_size_w", self.crop_size_w.value())

    def set_crop_size_w_slider(self):
        self.crop_size_w.setValue(self.crop_size_w_spinBox_2.value())
        write_config("crop_size_w", self.crop_size_w_spinBox_2.value())

    def ref_frame(self):
        get_ref_frame(input=read_config("file_root"))
        myWin.color_completed_signal.emit()

    def open_colored_image(self): # 显示颜色参考帧
        path = 'results/inference_random_diverse_color/full_resolution_results'
        colored_image_list = sorted(glob.glob(os.path.join(path, '*.png')))
        idx = read_config("color_image_idx")
        color_image = colored_image_list[idx]
        image = cv2.imread(color_image)
        image_height, image_width = image.shape[0], image.shape[1]
        long_w_h = image_width - image_height
        if long_w_h >= 0:
            #  w > h，加宽 h
            image_show = cv2.copyMakeBorder(image, int(long_w_h / 2), int(long_w_h / 2), 0, 0,
                                            cv2.BORDER_CONSTANT, value=[0, 0, 0])
        else:
            image_show = cv2.copyMakeBorder(image, 0, 0, int(long_w_h / 2), int(long_w_h / 2),
                                            cv2.BORDER_CONSTANT, value=[0, 0, 0])
        if True:
            if len(image_show.shape) == 3:
                image_show = cv2.cvtColor(image_show, cv2.COLOR_BGR2RGB)
                video_img = QImage(image_show.data, image_show.shape[1], image_show.shape[0], QImage.Format_RGB888)
            elif len(image_show.shape) == 1:
                video_img = QImage(image_show.data, image_show.shape[1], image_show.shape[0], QImage.Format_Indexed8)
            else:
                video_img = QImage(image_show.data, image_show.shape[1], image_show.shape[0], QImage.Format_RGB888)
            self.colored_image_label.setPixmap(QPixmap(video_img))
            self.colored_image_label.setScaledContents(True)

    def color_image_right(self):
        idx = read_config("color_image_idx")
        if idx < 5:
            idx += 1
            write_config("color_image_idx", idx)

    def color_image_left(self):
        idx = read_config("color_image_idx")
        if idx > 0:
            idx -= 1
            write_config("color_image_idx", idx)

    def color_error(self):
        self.sub1_console_push_text(text='请先生成颜色参考')

    # 以下是通用的函数
    # 所有功能统一使用这一个保存路径，不需要区分功能
    def completed_frames_save_root(self, mode='default'):
        # 以当前时间为文件名
        current_time = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())
        # 存入json
        config_file = os.path.join('config', 'file.json')
        config = json.load(open(config_file, 'r', encoding='utf-8'))

        if mode == 'default':
            config["completed_frames_root"] = os.path.join("cache", "completed_frames", current_time)
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)
                print('本次操作完成后的存放地址为:', config["completed_frames_root"])
        elif mode == 'inp_opti':
            config["inp_opti_completed_frames_root"] = os.path.join("cache","inp_optimization_frames", current_time)
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)
                print('优化后视频帧的存放地址为:', config["inp_opti_completed_frames_root"])

    # 接收来自线程的信号，显示“已完成”
    def function_completed(self):
        self.sub1_console_push_text('已完成')
        # 设置主界面2个lineEdit的状态
        if myWin.player_flag == 0:
            myWin.set_text(serial=1)
            if myWin.inp_deno_deg_sr_inter_deold_stab_flags == [0, 0, 0, 0, 1, 0, 0]:
                myWin.set_text(serial=2_2)
            else:
                myWin.set_text(serial=20)
        elif myWin.player_flag == 1:
            myWin.set_text(serial=1)
            myWin.set_text(serial=20)
        elif myWin.player_flag == 2:
            myWin.set_text(serial=10)
            if myWin.inp_deno_deg_sr_inter_deold_stab_flags == [0, 0, 0, 0, 1, 0, 0]:
                myWin.set_text(serial=2_2)
            else:
                myWin.set_text(serial=2_1)

        # 赋值跟随路径
        config_file = os.path.join('config', 'file.json')
        # 读取json
        with open(config_file, 'r', encoding='utf-8') as f:
            file = json.loads(f.read())
        path = file['completed_frames_root']
        # 存入json
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        config["follow_root"] = path
        config_json = json.dumps(config, ensure_ascii=False, indent=2)
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_json)
        # 测试
        with open(config_file, 'r', encoding='utf-8') as f:
            file = json.loads(f.read())
        path = file['completed_frames_root']
        print('跟随路径保存完成：', path)
        # 跟随状态重置为0
        myWin.follow_flag = 0


class SubWindow_2(QDialog, Ui_Dialog_Mask):
    def __init__(self,):
        super().__init__()
        self.setupUi(self)
        # 模型等初始化 todo 可能由于加载模型的原因，导致mask窗口打开速度比较慢，之后修改
        self.mask_gen_init = mask_gen_init_Thread()
        prop_net, fuse_net, s2m_ctrl, fbrs_ctrl, masks, \
        num_objects, mem_freq, mem_profile = self.mask_gen_init.run()

        # 下面的赋值是为了方便，一些改变后的初始化
        self.prop_net_init = prop_net
        self.fuse_net_init = fuse_net
        self.mem_freq_init = mem_freq
        self.mem_profile_init = mem_profile

        with open(os.path.join('config', 'file.json'), 'r', encoding='utf-8') as f:
            file = json.loads(f.read())
        self.frames_for_masks_root = file['file_root']

        self.masks = masks  # None
        self.num_objects = num_objects  # 初始化时，num_objects = 1
        self.s2m_controller = s2m_ctrl  # “涂鸦”的方式，生成单帧中目标的mask的模型
        self.fbrs_controller = fbrs_ctrl  # “点击”的方式，生成单帧中目标的mask的模型

        # # 加载图片，如果单独在mask窗口打开图片，需要重新初始化
        # self.images = load_images(self.frames_for_masks_root)
        # self.processor = InferenceCore(prop_net, fuse_net, images_to_torch(self.images, device='cpu'),
        #                                num_objects, mem_freq=mem_freq, mem_profile=mem_profile)
        # #                                            mem_freq=5,        men_profile=0
        # # print(self.images.shape) ex:(80, 480, 810, 3)
        # self.num_frames, self.height, self.width = self.images.shape[:3]

        # 刚启动窗口时，需要初始化 1 次； 当 self.frames_for_masks_root 变化，或者 num_objects 变化时，需要重新初始化。
        self.processor_init(prop_net, fuse_net, num_objects, mem_freq, mem_profile)


        # IOU computation
        if self.masks is not None:
            self.ious = np.zeros(self.num_frames)
            self.iou_curve = []

        # 目标个数
        self.spinBox.valueChanged.connect(self.spinbox_change)

        # some buttons
        self.play_button.clicked.connect(self.on_play)
        #self.run_button = QPushButton('Propagate')
        self.run_button.clicked.connect(self.on_run)
        #self.commit_button = QPushButton('Commit')
        self.commit_button.clicked.connect(self.on_commit)

        #self.undo_button = QPushButton('Undo')
        self.undo_button.clicked.connect(self.on_undo)
        #self.reset_button = QPushButton('Reset Frame')
        self.reset_button.clicked.connect(self.on_reset)
        #self.save_button = QPushButton('Save')
        self.save_button.clicked.connect(self.save)

        # LCD  左下角显示当前帧和总帧数
        # self.lcd = QTextEdit()
        self.lcd.setReadOnly(True)
        # self.lcd.setMaximumHeight(28)
        # self.lcd.setMaximumWidth(120)
        self.lcd.setText('{: 4d} / {: 4d}'.format(0, self.num_frames - 1))

        # timeline slider  进度条，self.timer --> 改变self.tl_slider的值 --> 值的变化连接
        # self.tl_slider = QSlider(Qt.Horizontal)
        self.tl_slider.valueChanged.connect(self.tl_slide)
        self.tl_slider.setMinimum(0)
        self.tl_slider.setMaximum(self.num_frames - 1)
        self.tl_slider.setValue(0)
        self.tl_slider.setTickPosition(QSlider.TicksBelow)
        self.tl_slider.setTickInterval(1)

        # brush size slider  显示刷子宽度
        #self.brush_label = QLabel()
        #self.brush_label.setAlignment(Qt.AlignCenter)
        # self.brush_label.setMinimumWidth(100)
        # Free模式下，负责调节刷子的宽度
        # self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.valueChanged.connect(self.brush_slide)
        self.brush_slider.setMinimum(1)
        self.brush_slider.setMaximum(100)
        self.brush_slider.setValue(3)
        self.brush_slider.setTickPosition(QSlider.TicksBelow)
        self.brush_slider.setTickInterval(2)
        self.brush_slider.setMinimumWidth(300)

        # combobox  可以调整图像的亮度
        # self.combo = QComboBox(self)
        # self.combo.addItem("davis")
        # self.combo.addItem("fade")
        # self.combo.addItem("light")
        self.combo.currentTextChanged.connect(self.set_viz_mode)

        # Radio buttons for type of interactions  左下角的 3 个选择按钮
        self.curr_interaction = 'Click'
        self.curr_interaction_name = '点击'
        #self.interaction_group = QButtonGroup()
        #self.radio_fbrs = QRadioButton('Click')
        #self.radio_s2m = QRadioButton('Scribble')
        #self.radio_free = QRadioButton('Free')
        # self.interaction_group.addButton(self.radio_fbrs)
        # self.interaction_group.addButton(self.radio_s2m)
        # self.interaction_group.addButton(self.radio_free)
        self.radio_fbrs.toggled.connect(self.interaction_radio_clicked)
        self.radio_s2m.toggled.connect(self.interaction_radio_clicked)
        self.radio_free.toggled.connect(self.interaction_radio_clicked)
        self.radio_fbrs.toggle()

        # Main canvas -> QLabel
        #self.main_canvas = QLabel()
        self.main_canvas.setSizePolicy(QSizePolicy.Expanding,
                                       QSizePolicy.Expanding)
        self.main_canvas.setAlignment(Qt.AlignCenter)
        self.main_canvas.setMinimumSize(100, 100)

        self.main_canvas.mousePressEvent = self.on_press
        self.main_canvas.mouseMoveEvent = self.on_motion
        self.main_canvas.setMouseTracking(True)  # Required for all-time tracking
        self.main_canvas.mouseReleaseEvent = self.on_release

        # Minimap -> Also a QLbal
        # self.minimap = QLabel()
        self.minimap.setSizePolicy(QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        self.minimap.setAlignment(Qt.AlignTop)
        self.minimap.setMinimumSize(100, 100)

        # Zoom-in buttons
        #self.zoom_p_button = QPushButton('Zoom +')
        self.zoom_p_button.clicked.connect(self.on_zoom_plus)
        #self.zoom_m_button = QPushButton('Zoom -')
        self.zoom_m_button.clicked.connect(self.on_zoom_minus)
        #self.finish_local_button = QPushButton('Finish Local')
        self.finish_local_button.clicked.connect(self.on_finish_local)
        self.finish_local_button.setDisabled(True)

        # Console on the GUI
        #self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        #self.console.setMinimumHeight(100)
        #self.console.setMaximumHeight(100)

        # progress bar
        #self.progress = QProgressBar(self)
        #self.progress.setGeometry(0, 0, 300, 25)
        #self.progress.setMinimumWidth(300)
        self.progress.setValue(0)
        self.progress.setMinimum(0)
        self.progress.setMaximum(100)
        self.progress.setFormat('修复进度')
        self.progress.setStyleSheet("QProgressBar{color: black;}")
        self.progress.setAlignment(Qt.AlignCenter)

        # timer 用于播放的定时器
        self.timer = QTimer()
        self.timer.setSingleShot(False)
        self.timer.timeout.connect(self.on_time)

        # Local mode related states
        self.ctrl_key = False  # 键盘上的 ctrl按键
        self.in_local_mode = False  # 是否是局部模式
        self.local_bb = None
        self.local_interactions = {}
        self.this_local_interactions = []
        self.local_interaction = None

        # initialize visualization
        self.viz_mode = 'davis'
        self.current_mask = np.zeros((self.num_frames, self.height, self.width), dtype=np.uint8)
        self.vis_map = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.vis_alpha = np.zeros((self.height, self.width, 1), dtype=np.float32)
        self.brush_vis_map = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.brush_vis_alpha = np.zeros((self.height, self.width, 1), dtype=np.float32)
        self.vis_hist = deque(maxlen=100)
        self.cursur = 0
        self.on_showing = None

        # initialize local visualization (which is mostly unknown at this point)
        self.local_vis_map = None
        self.local_vis_alpha = None
        self.local_brush_vis_map = None
        self.local_brush_vis_alpha = None
        self.local_vis_hist = deque(maxlen=100)

        # Zoom parameters
        self.zoom_pixels = 150

        # initialize action
        self.interactions = {}
        self.interactions['interact'] = [[] for _ in range(self.num_frames)]  # 交互的内容存在这里
        self.interactions['annotated_frame'] = []
        self.this_frame_interactions = []
        self.interaction = None
        self.reset_this_interaction()
        self.pressed = False
        self.right_click = False
        self.ctrl_size = False
        self.current_object = 1
        self.last_ex = self.last_ey = 0

        # Objects shortcuts 直接初始化 3 个切换快捷键
        # for i in range(1, num_objects + 1):
        for i in range(1, 3 + 1):
            QShortcut(QKeySequence(str(i)), self).activated.connect(functools.partial(self.hit_number_key, i))

        # <- and -> shortcuts   键盘左键、右键
        QShortcut(QKeySequence(Qt.Key_Left), self).activated.connect(self.on_prev)
        QShortcut(QKeySequence(Qt.Key_Right), self).activated.connect(self.on_next)

        # Mask saving
        # QShortcut(QKeySequence('s'), self).activated.connect(self.save)
        # QShortcut(QKeySequence('l'), self).activated.connect(self.debug_pressed)

        self.interacted_mask = None

        self.show_current_frame()
        self.show()

        self.waiting_to_start = True
        self.global_timer = Timer().start()
        self.algo_timer = Timer()
        self.user_timer = Timer()
        # self.console_push_text('Initialized.')
        self.console_push_text('初始化')

    # 分割数目的改变，最高支持3个
    def spinbox_change(self):
        # 1
        self.num_objects = self.spinBox.value()
        print("self.num_objects", self.num_objects)
        # 2
        self.processor = InferenceCore(self.prop_net_init, self.fuse_net_init, images_to_torch(self.images, device='cpu'),
                                       self.num_objects, mem_freq=self.mem_freq_init, mem_profile=self.mem_profile_init)
        # # 3
        # # Objects shortcuts
        # for i in range(1, self.num_objects + 1):
        #     QShortcut(QKeySequence(str(i)), self).activated.connect(functools.partial(self.hit_number_key, i))

    # todo 后期增加：打开文件按钮，可以使用来进行初始化
    def processor_init(self, prop_net, fuse_net, num_objects, mem_freq, mem_profile):
        self.images = load_images(self.frames_for_masks_root)
        self.processor = InferenceCore(prop_net, fuse_net, images_to_torch(self.images, device='cpu'),
                                       num_objects, mem_freq=mem_freq, mem_profile=mem_profile)
        #                                            mem_freq=5,        men_profile=0
        # print(self.images.shape) ex:(80, 480, 810, 3)
        self.num_frames, self.height, self.width = self.images.shape[:3]

    def resizeEvent(self, event):
        self.show_current_frame()

    def save(self):
        folder_path = str(QFileDialog.getExistingDirectory(self, "请选择保存路径"))
        if folder_path != '':
            self.console_push_text('正在保存 mask 和 mask覆盖图')
            # todo 如果不想保存 overlay，可以注释掉
            mask_dir = path.join(folder_path, 'mask')
            # overlay_dir = path.join(folder_path, 'overlay')
            os.makedirs(mask_dir, exist_ok=True)
            # os.makedirs(overlay_dir, exist_ok=True)
            for i in range(self.num_frames):
                # Save mask
                mask = Image.fromarray(self.current_mask[i]).convert('P')
                mask.putpalette(palette)
                mask.save(os.path.join(mask_dir, '{:05d}.png'.format(i)))

                # # Save overlay
                # overlay = overlay_davis(self.images[i], self.current_mask[i])
                # overlay = Image.fromarray(overlay)
                # overlay.save(os.path.join(overlay_dir, '{:05d}.png'.format(i)))

            self.console_push_text('保存成功')
        else:
            self.console_push_text('未保存')

    def console_push_text(self, text):
        # text = '[A: %s, U: %s]: %s' % (self.algo_timer.format(), self.user_timer.format(), text)
        text = '--> %s' % (text)
        self.console.appendPlainText(text)
        self.console.moveCursor(QTextCursor.End)
        # print(text)

    def interaction_radio_clicked(self, event):
        self.last_interaction = self.curr_interaction
        if self.radio_s2m.isChecked():
            self.curr_interaction = 'Scribble'
            self.curr_interaction_name ='涂鸦'
            self.brush_size = 3
            self.brush_slider.setDisabled(True)
        elif self.radio_fbrs.isChecked():
            self.curr_interaction = 'Click'
            self.curr_interaction_name = '点击'
            self.brush_size = 3
            self.brush_slider.setDisabled(True)
        elif self.radio_free.isChecked():
            self.brush_slider.setDisabled(False)
            self.brush_slide()
            self.curr_interaction = 'Free'
            self.curr_interaction_name = '任意'
        if self.curr_interaction == 'Scribble':
            self.commit_button.setEnabled(True)
        else:
            self.commit_button.setEnabled(False)

        # if self.last_interaction != self.curr_interaction:
        # self.console_push_text('Interaction changed to ' + self.curr_interaction + '.')

    def compose_current_im(self):
        if self.in_local_mode:
            if self.viz_mode == 'fade':
                self.viz = overlay_davis_fade(self.local_np_im, self.local_np_mask)
            elif self.viz_mode == 'davis':
                self.viz = overlay_davis(self.local_np_im, self.local_np_mask)
            elif self.viz_mode == 'light':
                self.viz = overlay_davis(self.local_np_im, self.local_np_mask, 0.9)
            else:
                raise NotImplementedError
        else:
            if self.viz_mode == 'fade':
                self.viz = overlay_davis_fade(self.images[self.cursur], self.current_mask[self.cursur])
            elif self.viz_mode == 'davis':
                self.viz = overlay_davis(self.images[self.cursur], self.current_mask[self.cursur])
            elif self.viz_mode == 'light':
                self.viz = overlay_davis(self.images[self.cursur], self.current_mask[self.cursur], 0.9)
            else:
                raise NotImplementedError

    def update_interact_vis(self):
        # Update the interactions without re-computing the overlay
        height, width, channel = self.viz.shape
        bytesPerLine = 3 * width

        if self.in_local_mode:
            vis_map = self.local_vis_map
            vis_alpha = self.local_vis_alpha
            brush_vis_map = self.local_brush_vis_map
            brush_vis_alpha = self.local_brush_vis_alpha
        else:
            vis_map = self.vis_map
            vis_alpha = self.vis_alpha
            brush_vis_map = self.brush_vis_map
            brush_vis_alpha = self.brush_vis_alpha

        self.viz_with_stroke = self.viz * (1 - vis_alpha) + vis_map * vis_alpha
        self.viz_with_stroke = self.viz_with_stroke * (1 - brush_vis_alpha) + brush_vis_map * brush_vis_alpha
        self.viz_with_stroke = self.viz_with_stroke.astype(np.uint8)

        qImg = QImage(self.viz_with_stroke.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.main_canvas.setPixmap(QPixmap(qImg.scaled(self.main_canvas.size(),
                                                       Qt.KeepAspectRatio, Qt.FastTransformation)))

        self.main_canvas_size = self.main_canvas.size()
        self.image_size = qImg.size()

    def update_minimap(self):
        # Limit it within the valid range
        if self.in_local_mode:
            if self.minimap_in_local_drawn:
                # Do not redraw
                return
            self.minimap_in_local_drawn = True
            patch = self.minimap_in_local.astype(np.uint8)
        else:
            ex, ey = self.last_ex, self.last_ey
            r = self.zoom_pixels // 2
            ex = int(round(max(r, min(self.width - r, ex))))
            ey = int(round(max(r, min(self.height - r, ey))))

            patch = self.viz_with_stroke[ey - r:ey + r, ex - r:ex + r, :].astype(np.uint8)

        height, width, channel = patch.shape
        bytesPerLine = 3 * width
        qImg = QImage(patch.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.minimap.setPixmap(QPixmap(qImg.scaled(self.minimap.size(),
                                                   Qt.KeepAspectRatio, Qt.FastTransformation)))

    def show_current_frame(self):
        # Re-compute overlay and show the image
        # 图片和mask融合
        self.compose_current_im()
        # 更新图片
        self.update_interact_vis()
        self.update_minimap()
        self.lcd.setText('{: 3d} / {: 3d}'.format(self.cursur, self.num_frames - 1))
        self.tl_slider.setValue(self.cursur)

    def get_scaled_pos(self, x, y):
        # Un-scale and un-pad the label coordinates into image coordinates
        oh, ow = self.image_size.height(), self.image_size.width()
        nh, nw = self.main_canvas_size.height(), self.main_canvas_size.width()

        h_ratio = nh / oh
        w_ratio = nw / ow
        dominate_ratio = min(h_ratio, w_ratio)

        # Solve scale
        x /= dominate_ratio
        y /= dominate_ratio

        # Solve padding
        fh, fw = nh / dominate_ratio, nw / dominate_ratio
        x -= (fw - ow) / 2
        y -= (fh - oh) / 2

        if self.in_local_mode:
            x = max(0, min(self.local_width - 1, x))
            y = max(0, min(self.local_height - 1, y))
        else:
            x = max(0, min(self.width - 1, x))
            y = max(0, min(self.height - 1, y))

        # return int(round(x)), int(round(y))
        return x, y

    def clear_visualization(self):
        if self.in_local_mode:
            self.local_vis_map.fill(0)
            self.local_vis_alpha.fill(0)
            self.local_vis_hist.clear()
            self.local_vis_hist.append((self.local_vis_map.copy(), self.local_vis_alpha.copy()))
        else:
            self.vis_map.fill(0)
            self.vis_alpha.fill(0)
            self.vis_hist.clear()
            self.vis_hist.append((self.vis_map.copy(), self.vis_alpha.copy()))

    def reset_this_interaction(self):
        self.complete_interaction()
        self.clear_visualization()
        if self.in_local_mode:
            self.local_interaction = None
            self.local_interactions['interact'] = self.local_interactions['interact'][:1]
        else:
            self.interaction = None
            self.this_frame_interactions = []
        self.undo_button.setDisabled(True)
        if self.fbrs_controller is not None:
            self.fbrs_controller.unanchor()

    def set_viz_mode(self):
        if self.combo.currentIndex() == 1:
            self.viz_mode = 'fade'
        elif self.combo.currentIndex() == 0:
            self.viz_mode = 'davis'
        elif self.combo.currentIndex() == 2:
            self.viz_mode = 'light'
        else:
            pass
        # self.viz_mode = self.combo.currentText()
        # print(self.combo.currentIndex(), self.combo.currentText())
        self.show_current_frame()

    # self.tl_slider 值的改变，和这个函数进行绑定，
    def tl_slide(self):
        if self.waiting_to_start:
            self.waiting_to_start = False
            self.algo_timer.start()
            self.user_timer.start()
            # self.console_push_text('Timers started.')

        self.reset_this_interaction()
        self.cursur = self.tl_slider.value()
        self.show_current_frame()

    # Free模式下，刷子宽度的调节
    def brush_slide(self):
        self.brush_size = self.brush_slider.value()
        self.brush_label.setText('宽度: %d' % self.brush_size)
        try:
            if type(self.interaction) == FreeInteraction:
                self.interaction.set_size(self.brush_size)
        except AttributeError:
            # Initialization, forget about it
            pass

    def progress_step_cb(self):
        self.progress_num += 1
        ratio = self.progress_num / self.progress_max
        self.progress.setValue(int(ratio * 100))
        self.progress.setFormat('%2.1f%%' % (ratio * 100))
        QApplication.processEvents()

    def progress_total_cb(self, total):
        self.progress_max = total
        self.progress_num = -1
        self.progress_step_cb()

    def on_run(self):
        self.user_timer.pause()
        if self.interacted_mask is None:
            # self.console_push_text('Cannot propagate! No interacted mask!')
            self.console_push_text('不能生成!没有交互的mask!')
            return

        # self.console_push_text('Propagation started.')
        self.console_push_text('开始生成')
        # self.interacted_mask = torch.softmax(self.interacted_mask*1000, dim=0)
        self.current_mask = self.processor.interact(self.interacted_mask, self.cursur,
                                                    self.progress_total_cb, self.progress_step_cb)
        self.interacted_mask = None
        # clear scribble and reset
        self.show_current_frame()
        self.reset_this_interaction()
        # self.progress.setFormat('Idle')
        self.progress.setFormat('空闲')
        self.progress.setValue(0)
        # self.console_push_text('Propagation finished!')
        self.console_push_text('生成完成')
        self.user_timer.start()

    def on_commit(self):
        self.complete_interaction()
        self.update_interacted_mask()

    def on_prev(self):
        # self.tl_slide will trigger on setValue
        self.cursur = max(0, self.cursur - 1)
        self.tl_slider.setValue(self.cursur)

    def on_next(self):
        # self.tl_slide will trigger on setValue
        self.cursur = min(self.cursur + 1, self.num_frames - 1)
        self.tl_slider.setValue(self.cursur)

    def on_time(self):
        self.cursur += 1
        if self.cursur > self.num_frames - 1:
            self.cursur = 0
        self.tl_slider.setValue(self.cursur)

    def on_play(self):
        if self.timer.isActive():
            self.timer.stop()
        else:
            # self.timer.start(1000 / 25)
            self.timer.start(40)

    def on_undo(self):
        if self.in_local_mode:
            if self.local_interaction is None:
                if len(self.local_interactions['interact']) > 1:
                    self.local_interactions['interact'] = self.local_interactions['interact'][:-1]
                else:
                    self.reset_this_interaction()
                self.local_interacted_mask = self.local_interactions['interact'][-1].predict()
            else:
                if self.local_interaction.can_undo():
                    self.local_interacted_mask = self.local_interaction.undo()
                else:
                    if len(self.local_interactions['interact']) > 1:
                        self.local_interaction = None
                    else:
                        self.reset_this_interaction()
                    self.local_interacted_mask = self.local_interactions['interact'][-1].predict()

            # Update visualization
            if len(self.local_vis_hist) > 0:
                # Might be empty if we are undoing the entire interaction
                self.local_vis_map, self.local_vis_alpha = self.local_vis_hist.pop()
        else:
            if self.interaction is None:
                if len(self.this_frame_interactions) > 1:
                    self.this_frame_interactions = self.this_frame_interactions[:-1]
                    self.interacted_mask = self.this_frame_interactions[-1].predict()
                else:
                    self.reset_this_interaction()
                    self.interacted_mask = self.processor.prob[:, self.cursur].clone()
            else:
                if self.interaction.can_undo():
                    self.interacted_mask = self.interaction.undo()
                else:
                    if len(self.this_frame_interactions) > 0:
                        self.interaction = None
                        self.interacted_mask = self.this_frame_interactions[-1].predict()
                    else:
                        self.reset_this_interaction()
                        self.interacted_mask = self.processor.prob[:, self.cursur].clone()

            # Update visualization
            if len(self.vis_hist) > 0:
                # Might be empty if we are undoing the entire interaction
                self.vis_map, self.vis_alpha = self.vis_hist.pop()

        # Commit changes
        self.update_interacted_mask()

    def on_reset(self):
        # DO not edit prob -- we still need the mask diff
        self.processor.masks[self.cursur].zero_()
        self.processor.np_masks[self.cursur].fill(0)
        self.current_mask[self.cursur].fill(0)
        self.reset_this_interaction()
        self.show_current_frame()

    def on_zoom_plus(self):
        self.zoom_pixels -= 25
        self.zoom_pixels = max(50, self.zoom_pixels)
        self.update_minimap()

    def on_zoom_minus(self):
        self.zoom_pixels += 25
        self.zoom_pixels = min(self.zoom_pixels, 300)
        self.update_minimap()

    def set_navi_enable(self, boolean):
        self.zoom_p_button.setEnabled(boolean)
        self.zoom_m_button.setEnabled(boolean)
        self.run_button.setEnabled(boolean)
        self.tl_slider.setEnabled(boolean)
        self.play_button.setEnabled(boolean)
        self.lcd.setEnabled(boolean)

    def on_finish_local(self):
        self.complete_interaction()
        self.finish_local_button.setDisabled(True)
        self.in_local_mode = False
        self.set_navi_enable(True)

        # Push the combined local interactions as a global interaction
        if len(self.this_frame_interactions) > 0:
            prev_soft_mask = self.this_frame_interactions[-1].out_prob
        else:
            prev_soft_mask = self.processor.prob[1:, self.cursur]
        image = self.processor.images[:, self.cursur]

        self.interaction = LocalInteraction(
            image, prev_soft_mask, (self.height, self.width), self.local_bb,
            self.local_interactions['interact'][-1].out_prob,
            self.processor.pad, self.local_pad
        )
        self.interaction.storage = self.local_interactions
        self.interacted_mask = self.interaction.predict()
        self.complete_interaction()
        self.update_interacted_mask()
        self.show_current_frame()

        # self.console_push_text('Finished local control.')
        self.console_push_text('局部操作结束')

    def hit_number_key(self, number):
        # number: 想要切换到的目标序号；  self.current_object：当前所选的目标
        if number == self.current_object:
            return
        # self.current_object = number
        # if self.fbrs_controller is not None:
        #     self.fbrs_controller.unanchor()
        # self.console_push_text('Current object changed to %d!' % number)
        if number <= self.num_objects:
            self.current_object = number
            if self.fbrs_controller is not None:
                self.fbrs_controller.unanchor()
            # self.console_push_text('Current object changed to %d!' % number)
            self.console_push_text('正在操作第 %d个目标!' % number)
        else:
            # print("请增大目标个数")
            self.console_push_text('请增大目标个数')

        self.clear_brush()
        self.vis_brush(self.last_ex, self.last_ey)
        self.update_interact_vis()
        self.show_current_frame()

    def clear_brush(self):
        self.brush_vis_map.fill(0)
        self.brush_vis_alpha.fill(0)
        if self.local_brush_vis_map is not None:
            self.local_brush_vis_map.fill(0)
            self.local_brush_vis_alpha.fill(0)

    def vis_brush(self, ex, ey):
        if self.ctrl_key:
            # Visualize the control region
            lx = int(round(min(self.local_start[0], ex)))
            ux = int(round(max(self.local_start[0], ex)))
            ly = int(round(min(self.local_start[1], ey)))
            uy = int(round(max(self.local_start[1], ey)))
            self.brush_vis_map = cv2.rectangle(self.brush_vis_map, (lx, ly), (ux, uy),
                                               (128, 255, 128), thickness=-1)
            self.brush_vis_alpha = cv2.rectangle(self.brush_vis_alpha, (lx, ly), (ux, uy),
                                                 0.5, thickness=-1)
        else:
            # Visualize the brush (yeah I know)
            if self.in_local_mode:
                self.local_brush_vis_map = cv2.circle(self.local_brush_vis_map,
                                                      (int(round(ex)), int(round(ey))), self.brush_size // 2 + 1,
                                                      color_map[self.current_object], thickness=-1)
                self.local_brush_vis_alpha = cv2.circle(self.local_brush_vis_alpha,
                                                        (int(round(ex)), int(round(ey))), self.brush_size // 2 + 1, 0.5,
                                                        thickness=-1)
            else:
                self.brush_vis_map = cv2.circle(self.brush_vis_map,
                                                (int(round(ex)), int(round(ey))), self.brush_size // 2 + 1,
                                                color_map[self.current_object], thickness=-1)
                self.brush_vis_alpha = cv2.circle(self.brush_vis_alpha,
                                                  (int(round(ex)), int(round(ey))), self.brush_size // 2 + 1, 0.5,
                                                  thickness=-1)

    def enter_local_control(self):
        self.in_local_mode = True
        lx = int(round(min(self.local_start[0], self.local_end[0])))
        ux = int(round(max(self.local_start[0], self.local_end[0])))
        ly = int(round(min(self.local_start[1], self.local_end[1])))
        uy = int(round(max(self.local_start[1], self.local_end[1])))

        # Reset variables
        self.local_bb = (lx, ux, ly, uy)
        self.local_interactions = {}
        self.local_interactions['interact'] = []
        self.local_interaction = None

        # Initial info
        if len(self.this_local_interactions) == 0:
            prev_soft_mask = self.processor.prob[1:, self.cursur]
        else:
            prev_soft_mask = self.this_local_interactions[-1].out_prob
        self.local_interactions['bounding_box'] = self.local_bb
        self.local_interactions['cursur'] = self.cursur
        init_interaction = CropperInteraction(self.processor.images[:, self.cursur],
                                              prev_soft_mask, self.processor.pad, self.local_bb)
        self.local_interactions['interact'].append(init_interaction)

        self.local_interacted_mask = init_interaction.out_mask
        self.local_torch_im = init_interaction.im_crop
        self.local_np_im = self.images[self.cursur][ly:uy + 1, lx:ux + 1, :]
        self.local_pad = init_interaction.pad

        # initialize the local visualization maps
        h, w = init_interaction.h, init_interaction.w
        self.local_vis_map = np.zeros((h, w, 3), dtype=np.uint8)
        self.local_vis_alpha = np.zeros((h, w, 1), dtype=np.float32)
        self.local_brush_vis_map = np.zeros((h, w, 3), dtype=np.uint8)
        self.local_brush_vis_alpha = np.zeros((h, w, 1), dtype=np.float32)
        self.local_vis_hist = deque(maxlen=100)
        self.local_height, self.local_width = h, w

        # Refresh self.viz
        self.minimap_in_local_drawn = False
        self.minimap_in_local = self.viz_with_stroke
        self.update_interacted_mask()
        self.finish_local_button.setEnabled(True)
        self.undo_button.setEnabled(False)
        self.set_navi_enable(False)

        # self.console_push_text('Entered local control.')
        self.console_push_text('开始局部操作')

    def on_press(self, event):
        if self.waiting_to_start:
            self.waiting_to_start = False
            self.algo_timer.start()
            self.user_timer.start()
            # self.console_push_text('Timers started.')

        self.user_timer.pause()
        ex, ey = self.get_scaled_pos(event.x(), event.y())
        # Check for ctrl key   ctrl+按住左键，选中一个区域，松开后会放大
        modifiers = QApplication.keyboardModifiers()
        if not self.in_local_mode and modifiers == QtCore.Qt.ControlModifier:
            # Start specifying the local mode
            self.ctrl_key = True
        else:
            self.ctrl_key = False

        self.pressed = True
        self.right_click = (event.button() != 1)
        # Push last vis map into history
        if self.in_local_mode:
            self.local_vis_hist.append((self.local_vis_map.copy(), self.local_vis_alpha.copy()))
        else:
            self.vis_hist.append((self.vis_map.copy(), self.vis_alpha.copy()))
        if self.ctrl_key:
            # Wrap up the last interaction
            self.complete_interaction()
            # Labeling a local control field
            self.local_start = ex, ey
        else:
            # Ordinary interaction (might be in local mode)
            if self.in_local_mode:
                if self.local_interaction is None:
                    prev_soft_mask = self.local_interactions['interact'][-1].out_prob
                else:
                    prev_soft_mask = self.local_interaction.out_prob
                prev_hard_mask = self.local_max_mask
                image = self.local_torch_im
                h, w = self.local_height, self.local_width
            else:
                if self.interaction is None:
                    if len(self.this_frame_interactions) > 0:
                        prev_soft_mask = self.this_frame_interactions[-1].out_prob
                    else:
                        prev_soft_mask = self.processor.prob[1:, self.cursur]
                else:
                    # Not used if the previous interaction is still valid
                    # Don't worry about stacking effects here
                    prev_soft_mask = self.interaction.out_prob
                prev_hard_mask = self.processor.masks[self.cursur]
                image = self.processor.images[:, self.cursur]
                h, w = self.height, self.width

            last_interaction = self.local_interaction if self.in_local_mode else self.interaction
            new_interaction = None
            if self.curr_interaction == 'Scribble':
                if last_interaction is None or type(last_interaction) != ScribbleInteraction:
                    self.complete_interaction()
                    new_interaction = ScribbleInteraction(image, prev_hard_mask, (h, w),
                                                          self.s2m_controller, self.num_objects)
            elif self.curr_interaction == 'Free':
                if last_interaction is None or type(last_interaction) != FreeInteraction:
                    self.complete_interaction()
                    if self.in_local_mode:
                        new_interaction = FreeInteraction(image, prev_soft_mask, (h, w),
                                                          self.num_objects, self.local_pad)
                    else:
                        new_interaction = FreeInteraction(image, prev_soft_mask, (h, w),
                                                          self.num_objects, self.processor.pad)
                    new_interaction.set_size(self.brush_size)
            elif self.curr_interaction == 'Click':
                if (last_interaction is None or type(last_interaction) != ClickInteraction
                        or last_interaction.tar_obj != self.current_object):
                    self.complete_interaction()
                    self.fbrs_controller.unanchor()
                    new_interaction = ClickInteraction(image, prev_soft_mask, (h, w),
                                                       self.fbrs_controller, self.current_object, self.processor.pad)

            if new_interaction is not None:
                if self.in_local_mode:
                    self.local_interaction = new_interaction
                else:
                    self.interaction = new_interaction

        # Just motion it as the first step
        self.on_motion(event)
        self.user_timer.start()

    def on_motion(self, event):
        ex, ey = self.get_scaled_pos(event.x(), event.y())
        self.last_ex, self.last_ey = ex, ey
        self.clear_brush()
        # Visualize
        self.vis_brush(ex, ey)
        if self.pressed:
            if not self.ctrl_key:
                if self.curr_interaction == 'Scribble' or self.curr_interaction == 'Free':
                    obj = 0 if self.right_click else self.current_object
                    # Actually draw it if dragging
                    if self.in_local_mode:
                        self.local_vis_map, self.local_vis_alpha = self.local_interaction.push_point(
                            ex, ey, obj, (self.local_vis_map, self.local_vis_alpha)
                        )
                    else:
                        self.vis_map, self.vis_alpha = self.interaction.push_point(
                            ex, ey, obj, (self.vis_map, self.vis_alpha)
                        )
        self.update_interact_vis()
        self.update_minimap()

    def update_interacted_mask(self):
        if self.in_local_mode:
            self.local_max_mask = torch.argmax(self.local_interacted_mask, 0)
            max_mask = unpad_3dim(self.local_max_mask, self.local_pad)
            self.local_np_mask = (max_mask.detach().cpu().numpy()[0]).astype(np.uint8)
        else:
            self.processor.update_mask_only(self.interacted_mask, self.cursur)
            self.current_mask[self.cursur] = self.processor.np_masks[self.cursur]
        self.show_current_frame()

    def complete_interaction(self):
        if self.in_local_mode:
            if self.local_interaction is not None:
                self.clear_visualization()
                self.local_interactions['interact'].append(self.local_interaction)
                self.local_interaction = None
                self.undo_button.setDisabled(False)
        else:
            if self.interaction is not None:
                self.clear_visualization()
                self.interactions['annotated_frame'].append(self.cursur)
                self.interactions['interact'][self.cursur].append(self.interaction)
                self.this_frame_interactions.append(self.interaction)
                self.interaction = None
                self.undo_button.setDisabled(False)

    def on_release(self, event):
        self.user_timer.pause()
        ex, ey = self.get_scaled_pos(event.x(), event.y())
        if self.ctrl_key:
            # Enter local control mode
            self.clear_visualization()
            self.local_end = ex, ey
            self.enter_local_control()
        else:
            # self.console_push_text('Interaction %s at frame %d.' % (self.curr_interaction, self.cursur))
            self.console_push_text('第%d帧，交互方式：%s' % (self.cursur, self.curr_interaction_name))
            # Ordinary interaction (might be in local mode)
            if self.in_local_mode:
                interaction = self.local_interaction
            else:
                interaction = self.interaction

            if self.curr_interaction == 'Scribble' or self.curr_interaction == 'Free':
                self.on_motion(event)
                interaction.end_path()
                if self.curr_interaction == 'Free':
                    self.clear_visualization()
            elif self.curr_interaction == 'Click':
                ex, ey = self.get_scaled_pos(event.x(), event.y())
                if self.in_local_mode:
                    self.local_vis_map, self.local_vis_alpha = interaction.push_point(ex, ey,
                                                                                      self.right_click, (
                                                                                      self.local_vis_map,
                                                                                      self.local_vis_alpha))
                else:
                    self.vis_map, self.vis_alpha = interaction.push_point(ex, ey,
                                                                          self.right_click,
                                                                          (self.vis_map, self.vis_alpha))

            if self.in_local_mode:
                self.local_interacted_mask = interaction.predict()
            else:
                self.interacted_mask = interaction.predict()
            self.update_interacted_mask()

        self.pressed = self.ctrl_key = self.right_click = False
        self.undo_button.setDisabled(False)
        self.user_timer.start()

    def debug_pressed(self):
        self.debug_mask, self.interacted_mask = self.interacted_mask, self.debug_mask

        self.processor.update_mask_only(self.interacted_mask, self.cursur)
        self.current_mask[self.cursur] = self.processor.np_masks[self.cursur]
        self.show_current_frame()

    def wheelEvent(self, event):
        ex, ey = self.get_scaled_pos(event.x(), event.y())
        if self.curr_interaction == 'Free':
            self.brush_slider.setValue(self.brush_slider.value() + event.angleDelta().y() // 30)
        self.clear_brush()
        self.vis_brush(ex, ey)
        self.update_interact_vis()
        self.update_minimap()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # subWin1 = SubWindow_1()
    myWin = MainWindow()
    myWin.show()
    # myWin.showMaximized()
    sys.exit(app.exec_())
