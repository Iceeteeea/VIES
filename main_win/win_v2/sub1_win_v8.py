# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'sub1_win_v8.ui'
#
# Created by: PyQt5 UI code generator 5.12
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog_tool(object):
    def setupUi(self, Dialog_tool):
        Dialog_tool.setObjectName("Dialog_tool")
        Dialog_tool.resize(472, 428)
        self.sub1_console_plainTextEdit = QtWidgets.QPlainTextEdit(Dialog_tool)
        self.sub1_console_plainTextEdit.setGeometry(QtCore.QRect(140, 350, 301, 61))
        self.sub1_console_plainTextEdit.setReadOnly(True)
        self.sub1_console_plainTextEdit.setObjectName("sub1_console_plainTextEdit")
        self.sub1_start_pushButton = QtWidgets.QPushButton(Dialog_tool)
        self.sub1_start_pushButton.setGeometry(QtCore.QRect(80, 360, 51, 31))
        self.sub1_start_pushButton.setObjectName("sub1_start_pushButton")
        self.sub1_tools_tabWidget = QtWidgets.QTabWidget(Dialog_tool)
        self.sub1_tools_tabWidget.setGeometry(QtCore.QRect(10, 10, 431, 331))
        self.sub1_tools_tabWidget.setObjectName("sub1_tools_tabWidget")
        self.sub1_deno_tab = QtWidgets.QWidget()
        self.sub1_deno_tab.setObjectName("sub1_deno_tab")
        self.denoise_Slider = QtWidgets.QSlider(self.sub1_deno_tab)
        self.denoise_Slider.setGeometry(QtCore.QRect(180, 40, 160, 22))
        self.denoise_Slider.setOrientation(QtCore.Qt.Horizontal)
        self.denoise_Slider.setObjectName("denoise_Slider")
        self.label = QtWidgets.QLabel(self.sub1_deno_tab)
        self.label.setGeometry(QtCore.QRect(80, 40, 71, 21))
        self.label.setObjectName("label")
        self.max_num_fr_per_seq_spinBox = QtWidgets.QSpinBox(self.sub1_deno_tab)
        self.max_num_fr_per_seq_spinBox.setGeometry(QtCore.QRect(170, 110, 42, 22))
        self.max_num_fr_per_seq_spinBox.setObjectName("max_num_fr_per_seq_spinBox")
        self.label_7 = QtWidgets.QLabel(self.sub1_deno_tab)
        self.label_7.setGeometry(QtCore.QRect(220, 110, 111, 16))
        self.label_7.setObjectName("label_7")
        self.sub1_tools_tabWidget.addTab(self.sub1_deno_tab, "")
        self.sub1_deg_tab = QtWidgets.QWidget()
        self.sub1_deg_tab.setObjectName("sub1_deg_tab")
        self.label_6 = QtWidgets.QLabel(self.sub1_deg_tab)
        self.label_6.setGeometry(QtCore.QRect(10, 30, 251, 21))
        self.label_6.setObjectName("label_6")
        self.mindim_Slider = QtWidgets.QSlider(self.sub1_deg_tab)
        self.mindim_Slider.setGeometry(QtCore.QRect(10, 70, 160, 22))
        self.mindim_Slider.setOrientation(QtCore.Qt.Horizontal)
        self.mindim_Slider.setObjectName("mindim_Slider")
        self.sub1_tools_tabWidget.addTab(self.sub1_deg_tab, "")
        self.sub1_stab_tab = QtWidgets.QWidget()
        self.sub1_stab_tab.setObjectName("sub1_stab_tab")
        self.sub1_tools_tabWidget.addTab(self.sub1_stab_tab, "")
        self.sub1_inp_tab = QtWidgets.QWidget()
        self.sub1_inp_tab.setObjectName("sub1_inp_tab")
        self.label_2 = QtWidgets.QLabel(self.sub1_inp_tab)
        self.label_2.setGeometry(QtCore.QRect(20, 20, 71, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.sub1_comboBox = QtWidgets.QComboBox(self.sub1_inp_tab)
        self.sub1_comboBox.setGeometry(QtCore.QRect(110, 20, 81, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.sub1_comboBox.setFont(font)
        self.sub1_comboBox.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.sub1_comboBox.setObjectName("sub1_comboBox")
        self.sub1_comboBox.addItem("")
        self.sub1_comboBox.addItem("")
        self.sub1_comboBox.addItem("")
        self.label_3 = QtWidgets.QLabel(self.sub1_inp_tab)
        self.label_3.setGeometry(QtCore.QRect(20, 80, 71, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.sub1_open_mask_pushButton = QtWidgets.QPushButton(self.sub1_inp_tab)
        self.sub1_open_mask_pushButton.setGeometry(QtCore.QRect(110, 80, 81, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.sub1_open_mask_pushButton.setFont(font)
        self.sub1_open_mask_pushButton.setObjectName("sub1_open_mask_pushButton")
        self.sub1_mask_gen_pushButton = QtWidgets.QPushButton(self.sub1_inp_tab)
        self.sub1_mask_gen_pushButton.setGeometry(QtCore.QRect(220, 80, 81, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.sub1_mask_gen_pushButton.setFont(font)
        self.sub1_mask_gen_pushButton.setObjectName("sub1_mask_gen_pushButton")
        self.line = QtWidgets.QFrame(self.sub1_inp_tab)
        self.line.setGeometry(QtCore.QRect(0, 130, 431, 20))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.label_4 = QtWidgets.QLabel(self.sub1_inp_tab)
        self.label_4.setGeometry(QtCore.QRect(20, 150, 51, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.sub1_inp_tab)
        self.label_5.setGeometry(QtCore.QRect(10, 210, 61, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.sub1_remove_opt_frames_itextEdit = QtWidgets.QTextEdit(self.sub1_inp_tab)
        self.sub1_remove_opt_frames_itextEdit.setGeometry(QtCore.QRect(80, 190, 321, 70))
        self.sub1_remove_opt_frames_itextEdit.setReadOnly(True)
        self.sub1_remove_opt_frames_itextEdit.setObjectName("sub1_remove_opt_frames_itextEdit")
        self.sub1_opti_add_pushButton = QtWidgets.QPushButton(self.sub1_inp_tab)
        self.sub1_opti_add_pushButton.setGeometry(QtCore.QRect(140, 270, 41, 23))
        self.sub1_opti_add_pushButton.setObjectName("sub1_opti_add_pushButton")
        self.sub1_opti_dec_pushButton = QtWidgets.QPushButton(self.sub1_inp_tab)
        self.sub1_opti_dec_pushButton.setGeometry(QtCore.QRect(210, 270, 41, 23))
        self.sub1_opti_dec_pushButton.setObjectName("sub1_opti_dec_pushButton")
        self.sub1_inp_opt_choose_radioButton = QtWidgets.QRadioButton(self.sub1_inp_tab)
        self.sub1_inp_opt_choose_radioButton.setGeometry(QtCore.QRect(80, 160, 31, 21))
        self.sub1_inp_opt_choose_radioButton.setText("")
        self.sub1_inp_opt_choose_radioButton.setObjectName("sub1_inp_opt_choose_radioButton")
        self.sub1_opti_clear_pushButton = QtWidgets.QPushButton(self.sub1_inp_tab)
        self.sub1_opti_clear_pushButton.setGeometry(QtCore.QRect(280, 270, 41, 23))
        self.sub1_opti_clear_pushButton.setObjectName("sub1_opti_clear_pushButton")
        self.sub1_tools_tabWidget.addTab(self.sub1_inp_tab, "")
        self.sub1_sr_tab = QtWidgets.QWidget()
        self.sub1_sr_tab.setObjectName("sub1_sr_tab")
        self.sub1_sr_comboBox = QtWidgets.QComboBox(self.sub1_sr_tab)
        self.sub1_sr_comboBox.setGeometry(QtCore.QRect(50, 40, 111, 21))
        self.sub1_sr_comboBox.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.sub1_sr_comboBox.setObjectName("sub1_sr_comboBox")
        self.sub1_sr_comboBox.addItem("")
        self.sub1_sr_comboBox.addItem("")
        self.sub1_sr_comboBox.addItem("")
        self.sub1_tools_tabWidget.addTab(self.sub1_sr_tab, "")
        self.sub1_inter_tab = QtWidgets.QWidget()
        self.sub1_inter_tab.setObjectName("sub1_inter_tab")
        self.sub1_intp_comboBox = QtWidgets.QComboBox(self.sub1_inter_tab)
        self.sub1_intp_comboBox.setGeometry(QtCore.QRect(160, 100, 111, 21))
        self.sub1_intp_comboBox.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.sub1_intp_comboBox.setObjectName("sub1_intp_comboBox")
        self.sub1_intp_comboBox.addItem("")
        self.sub1_intp_comboBox.addItem("")
        self.sub1_intp_comboBox.addItem("")
        self.intp_is4k_checkBox = QtWidgets.QCheckBox(self.sub1_inter_tab)
        self.intp_is4k_checkBox.setGeometry(QtCore.QRect(300, 170, 71, 16))
        self.intp_is4k_checkBox.setObjectName("intp_is4k_checkBox")
        self.label_12 = QtWidgets.QLabel(self.sub1_inter_tab)
        self.label_12.setGeometry(QtCore.QRect(190, 80, 54, 12))
        self.label_12.setObjectName("label_12")
        self.label_13 = QtWidgets.QLabel(self.sub1_inter_tab)
        self.label_13.setGeometry(QtCore.QRect(20, 170, 61, 16))
        self.label_13.setObjectName("label_13")
        self.save_video_Button = QtWidgets.QPushButton(self.sub1_inter_tab)
        self.save_video_Button.setGeometry(QtCore.QRect(180, 210, 75, 23))
        self.save_video_Button.setObjectName("save_video_Button")
        self.fps_label = QtWidgets.QLabel(self.sub1_inter_tab)
        self.fps_label.setGeometry(QtCore.QRect(85, 170, 54, 12))
        self.fps_label.setObjectName("fps_label")
        self.sub1_tools_tabWidget.addTab(self.sub1_inter_tab, "")
        self.sub1_deold_tab = QtWidgets.QWidget()
        self.sub1_deold_tab.setObjectName("sub1_deold_tab")
        self.crop_size_h = QtWidgets.QSlider(self.sub1_deold_tab)
        self.crop_size_h.setGeometry(QtCore.QRect(85, 210, 121, 22))
        self.crop_size_h.setOrientation(QtCore.Qt.Horizontal)
        self.crop_size_h.setObjectName("crop_size_h")
        self.crop_size_w = QtWidgets.QSlider(self.sub1_deold_tab)
        self.crop_size_w.setGeometry(QtCore.QRect(85, 260, 121, 22))
        self.crop_size_w.setOrientation(QtCore.Qt.Horizontal)
        self.crop_size_w.setObjectName("crop_size_w")
        self.label_10 = QtWidgets.QLabel(self.sub1_deold_tab)
        self.label_10.setGeometry(QtCore.QRect(0, 210, 81, 16))
        self.label_10.setObjectName("label_10")
        self.label_11 = QtWidgets.QLabel(self.sub1_deold_tab)
        self.label_11.setGeometry(QtCore.QRect(0, 260, 81, 16))
        self.label_11.setObjectName("label_11")
        self.crop_size_h_spinBox = QtWidgets.QSpinBox(self.sub1_deold_tab)
        self.crop_size_h_spinBox.setGeometry(QtCore.QRect(215, 210, 42, 22))
        self.crop_size_h_spinBox.setObjectName("crop_size_h_spinBox")
        self.crop_size_w_spinBox_2 = QtWidgets.QSpinBox(self.sub1_deold_tab)
        self.crop_size_w_spinBox_2.setGeometry(QtCore.QRect(215, 260, 42, 22))
        self.crop_size_w_spinBox_2.setObjectName("crop_size_w_spinBox_2")
        self.color_frame_button = QtWidgets.QPushButton(self.sub1_deold_tab)
        self.color_frame_button.setGeometry(QtCore.QRect(300, 230, 93, 28))
        self.color_frame_button.setObjectName("color_frame_button")
        self.colored_image_label = QtWidgets.QLabel(self.sub1_deold_tab)
        self.colored_image_label.setGeometry(QtCore.QRect(90, 20, 251, 171))
        self.colored_image_label.setStyleSheet("background-color: rgb(70, 52, 52);")
        self.colored_image_label.setText("")
        self.colored_image_label.setObjectName("colored_image_label")
        self.color_image_right_button = QtWidgets.QPushButton(self.sub1_deold_tab)
        self.color_image_right_button.setGeometry(QtCore.QRect(340, 100, 21, 23))
        self.color_image_right_button.setObjectName("color_image_right_button")
        self.color_image_left_button = QtWidgets.QPushButton(self.sub1_deold_tab)
        self.color_image_left_button.setGeometry(QtCore.QRect(70, 100, 21, 23))
        self.color_image_left_button.setObjectName("color_image_left_button")
        self.sub1_tools_tabWidget.addTab(self.sub1_deold_tab, "")
        self.sub1_follow_pushButton = QtWidgets.QPushButton(Dialog_tool)
        self.sub1_follow_pushButton.setGeometry(QtCore.QRect(10, 360, 51, 31))
        self.sub1_follow_pushButton.setObjectName("sub1_follow_pushButton")

        self.retranslateUi(Dialog_tool)
        self.sub1_tools_tabWidget.setCurrentIndex(3)
        QtCore.QMetaObject.connectSlotsByName(Dialog_tool)

    def retranslateUi(self, Dialog_tool):
        _translate = QtCore.QCoreApplication.translate
        Dialog_tool.setWindowTitle(_translate("Dialog_tool", "Dialog"))
        self.sub1_start_pushButton.setText(_translate("Dialog_tool", "开始"))
        self.label.setText(_translate("Dialog_tool", "去噪强度"))
        self.label_7.setText(_translate("Dialog_tool", "序列最大帧数"))
        self.sub1_tools_tabWidget.setTabText(self.sub1_tools_tabWidget.indexOf(self.sub1_deno_tab), _translate("Dialog_tool", "去噪"))
        self.label_6.setText(_translate("Dialog_tool", "Minimum edge dimension of the input video"))
        self.sub1_tools_tabWidget.setTabText(self.sub1_tools_tabWidget.indexOf(self.sub1_deg_tab), _translate("Dialog_tool", "去污损"))
        self.sub1_tools_tabWidget.setTabText(self.sub1_tools_tabWidget.indexOf(self.sub1_stab_tab), _translate("Dialog_tool", "稳相"))
        self.label_2.setText(_translate("Dialog_tool", "AI模型："))
        self.sub1_comboBox.setItemText(0, _translate("Dialog_tool", "极速版"))
        self.sub1_comboBox.setItemText(1, _translate("Dialog_tool", "普通版"))
        self.sub1_comboBox.setItemText(2, _translate("Dialog_tool", "增强版"))
        self.label_3.setText(_translate("Dialog_tool", "mask："))
        self.sub1_open_mask_pushButton.setText(_translate("Dialog_tool", "打开mask"))
        self.sub1_mask_gen_pushButton.setText(_translate("Dialog_tool", "mask生成"))
        self.label_4.setText(_translate("Dialog_tool", "优 化"))
        self.label_5.setText(_translate("Dialog_tool", "待优化帧"))
        self.sub1_opti_add_pushButton.setText(_translate("Dialog_tool", "+"))
        self.sub1_opti_dec_pushButton.setText(_translate("Dialog_tool", "-"))
        self.sub1_opti_clear_pushButton.setText(_translate("Dialog_tool", "clear"))
        self.sub1_tools_tabWidget.setTabText(self.sub1_tools_tabWidget.indexOf(self.sub1_inp_tab), _translate("Dialog_tool", "目标移除"))
        self.sub1_sr_comboBox.setItemText(0, _translate("Dialog_tool", "x2"))
        self.sub1_sr_comboBox.setItemText(1, _translate("Dialog_tool", "x4"))
        self.sub1_sr_comboBox.setItemText(2, _translate("Dialog_tool", "x8"))
        self.sub1_tools_tabWidget.setTabText(self.sub1_tools_tabWidget.indexOf(self.sub1_sr_tab), _translate("Dialog_tool", "超分"))
        self.sub1_intp_comboBox.setItemText(0, _translate("Dialog_tool", "x2"))
        self.sub1_intp_comboBox.setItemText(1, _translate("Dialog_tool", "x4"))
        self.sub1_intp_comboBox.setItemText(2, _translate("Dialog_tool", "x8"))
        self.intp_is4k_checkBox.setText(_translate("Dialog_tool", "4K视频"))
        self.label_12.setText(_translate("Dialog_tool", "插帧倍数"))
        self.label_13.setText(_translate("Dialog_tool", "插帧后fps:"))
        self.save_video_Button.setText(_translate("Dialog_tool", "save"))
        self.fps_label.setText(_translate("Dialog_tool", "50"))
        self.sub1_tools_tabWidget.setTabText(self.sub1_tools_tabWidget.indexOf(self.sub1_inter_tab), _translate("Dialog_tool", "插帧"))
        self.label_10.setText(_translate("Dialog_tool", "crop size h"))
        self.label_11.setText(_translate("Dialog_tool", "crop size w"))
        self.color_frame_button.setText(_translate("Dialog_tool", "生成颜色参考"))
        self.color_image_right_button.setText(_translate("Dialog_tool", ">"))
        self.color_image_left_button.setText(_translate("Dialog_tool", "<"))
        self.sub1_tools_tabWidget.setTabText(self.sub1_tools_tabWidget.indexOf(self.sub1_deold_tab), _translate("Dialog_tool", "上色"))
        self.sub1_follow_pushButton.setText(_translate("Dialog_tool", "跟随"))


