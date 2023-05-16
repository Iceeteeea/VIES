# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main_win_v10_1.ui'
#
# Created by: PyQt5 UI code generator 5.12
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1425, 868)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("../../image/icon.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setStyleSheet("#MainWindow{\n"
"    background-color: rgba(107, 107, 107,99);}\n"
"QSlider::groove:horizontal {\n"
"    height: 10px;\n"
"    border-radius: 5px;\n"
"    background-color: #dcdcdc;\n"
"}\n"
"\n"
"QSlider::handle:horizontal {\n"
"    width: 20px;\n"
"    margin: -5px 0;\n"
"    border-radius: 10px;\n"
"    background-color: rgb(255, 170, 0);\n"
"}\n"
"\n"
"QSlider::sub-page:horizontal {\n"
"    height: 10px;\n"
"    border-radius: 5px;\n"
"    background-color: #000;\n"
"}\n"
"QSlider::handle:hover {\n"
"    border: 1px solid #555;\n"
"    background-color: #000;\n"
"}\n"
"QPushButton:pressed{\n"
"    padding-top:5px;\n"
"    padding-left:5px;\n"
"}\n"
"\n"
"")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(11, 10, 1402, 801))
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout_2.addItem(spacerItem)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.horizontalLayout_5.addItem(spacerItem1)
        self.splitter = QtWidgets.QSplitter(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.splitter.sizePolicy().hasHeightForWidth())
        self.splitter.setSizePolicy(sizePolicy)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")
        self.player_groupBox_1 = QtWidgets.QGroupBox(self.splitter)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.player_groupBox_1.sizePolicy().hasHeightForWidth())
        self.player_groupBox_1.setSizePolicy(sizePolicy)
        self.player_groupBox_1.setMinimumSize(QtCore.QSize(669, 610))
        self.player_groupBox_1.setMaximumSize(QtCore.QSize(669, 610))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.player_groupBox_1.setFont(font)
        self.player_groupBox_1.setStyleSheet("color: rgb(255, 255, 255);")
        self.player_groupBox_1.setTitle("")
        self.player_groupBox_1.setObjectName("player_groupBox_1")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.player_groupBox_1)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.video_player_label_1 = QtWidgets.QLabel(self.player_groupBox_1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.video_player_label_1.sizePolicy().hasHeightForWidth())
        self.video_player_label_1.setSizePolicy(sizePolicy)
        self.video_player_label_1.setMinimumSize(QtCore.QSize(645, 586))
        self.video_player_label_1.setMaximumSize(QtCore.QSize(645, 586))
        self.video_player_label_1.setStyleSheet("background-color: rgba(107, 107, 107,99);")
        self.video_player_label_1.setText("")
        self.video_player_label_1.setObjectName("video_player_label_1")
        self.horizontalLayout_4.addWidget(self.video_player_label_1)
        self.player_groupBox_2 = QtWidgets.QGroupBox(self.splitter)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.player_groupBox_2.sizePolicy().hasHeightForWidth())
        self.player_groupBox_2.setSizePolicy(sizePolicy)
        self.player_groupBox_2.setMinimumSize(QtCore.QSize(669, 610))
        self.player_groupBox_2.setMaximumSize(QtCore.QSize(669, 610))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.player_groupBox_2.setFont(font)
        self.player_groupBox_2.setStyleSheet("color: rgb(255, 255, 255);")
        self.player_groupBox_2.setTitle("")
        self.player_groupBox_2.setObjectName("player_groupBox_2")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.player_groupBox_2)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.video_player_label_2 = QtWidgets.QLabel(self.player_groupBox_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.video_player_label_2.sizePolicy().hasHeightForWidth())
        self.video_player_label_2.setSizePolicy(sizePolicy)
        self.video_player_label_2.setMinimumSize(QtCore.QSize(645, 586))
        self.video_player_label_2.setMaximumSize(QtCore.QSize(645, 586))
        self.video_player_label_2.setStyleSheet("background-color: rgba(107, 107, 107,99);")
        self.video_player_label_2.setText("")
        self.video_player_label_2.setObjectName("video_player_label_2")
        self.verticalLayout_5.addWidget(self.video_player_label_2)
        self.horizontalLayout_5.addWidget(self.splitter)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.horizontalLayout_5.addItem(spacerItem2)
        self.verticalLayout_2.addLayout(self.horizontalLayout_5)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout_2.addItem(spacerItem3)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        spacerItem4 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_6.addItem(spacerItem4)
        self.frame = QtWidgets.QFrame(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame.sizePolicy().hasHeightForWidth())
        self.frame.setSizePolicy(sizePolicy)
        self.frame.setStyleSheet("#frame{\n"
"    background-color: rgba(107, 107, 107,99);\n"
" border-radius:5px;}\n"
"QPushButton{\n"
"    background-color:rgba(120, 120, 120,99);\n"
"    color:rgb(255, 255, 255);\n"
"    border-radius:5px;\n"
"}\n"
"QPushButton:pressed{\n"
"    padding-top:5px;\n"
"    padding-left:5px;\n"
"}\n"
"QLineEdit{ background-color:rgba(120, 120, 120,99);\n"
"    color:rgb(255, 255, 255);\n"
"    border-radius:5px;\n"
"}\n"
"\n"
"\n"
"")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.frame)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        spacerItem5 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.horizontalLayout_3.addItem(spacerItem5)
        self.start_stop_pushButton = QtWidgets.QPushButton(self.frame)
        self.start_stop_pushButton.setMinimumSize(QtCore.QSize(60, 60))
        self.start_stop_pushButton.setMaximumSize(QtCore.QSize(60, 60))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.start_stop_pushButton.setFont(font)
        self.start_stop_pushButton.setStyleSheet("background-color: rgba(0,0, 0, 0);\n"
"border-image:url(:/bg/run.png)")
        self.start_stop_pushButton.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/bg/运行.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.start_stop_pushButton.setIcon(icon1)
        self.start_stop_pushButton.setIconSize(QtCore.QSize(50, 50))
        self.start_stop_pushButton.setObjectName("start_stop_pushButton")
        self.horizontalLayout_3.addWidget(self.start_stop_pushButton)
        spacerItem6 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.horizontalLayout_3.addItem(spacerItem6)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame_SN_Slider = QtWidgets.QSlider(self.frame)
        self.frame_SN_Slider.setStyleSheet("QSlider::sub-page:horizontal {\n"
"     background-color: rgb(75, 75, 225)\n"
"     }\n"
"QSlider::handle:horizontal {\n"
"     background-color: rgb(255, 255, 255)\n"
"     }\n"
"QSlider::add-page:horizontal {\n"
"     background-color: rgb(212, 212, 212)\n"
"     }  \n"
"QSlider::handle:horizontal:hover {\n"
"     background-color:rgb(0, 150, 255)\n"
"     }")
        self.frame_SN_Slider.setOrientation(QtCore.Qt.Horizontal)
        self.frame_SN_Slider.setObjectName("frame_SN_Slider")
        self.verticalLayout.addWidget(self.frame_SN_Slider)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem7 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem7)
        self.previous_frame_pushButton = QtWidgets.QPushButton(self.frame)
        self.previous_frame_pushButton.setMinimumSize(QtCore.QSize(40, 40))
        self.previous_frame_pushButton.setMaximumSize(QtCore.QSize(40, 40))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.previous_frame_pushButton.setFont(font)
        self.previous_frame_pushButton.setObjectName("previous_frame_pushButton")
        self.horizontalLayout_2.addWidget(self.previous_frame_pushButton)
        spacerItem8 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem8)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.current_all_frames_lineEdit = QtWidgets.QLineEdit(self.frame)
        self.current_all_frames_lineEdit.setMinimumSize(QtCore.QSize(100, 40))
        self.current_all_frames_lineEdit.setMaximumSize(QtCore.QSize(100, 40))
        self.current_all_frames_lineEdit.setText("")
        self.current_all_frames_lineEdit.setAlignment(QtCore.Qt.AlignCenter)
        self.current_all_frames_lineEdit.setReadOnly(True)
        self.current_all_frames_lineEdit.setObjectName("current_all_frames_lineEdit")
        self.horizontalLayout.addWidget(self.current_all_frames_lineEdit)
        spacerItem9 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem9)
        self.choose_player_pushButton = QtWidgets.QPushButton(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.choose_player_pushButton.sizePolicy().hasHeightForWidth())
        self.choose_player_pushButton.setSizePolicy(sizePolicy)
        self.choose_player_pushButton.setMinimumSize(QtCore.QSize(100, 40))
        self.choose_player_pushButton.setMaximumSize(QtCore.QSize(100, 40))
        self.choose_player_pushButton.setFlat(False)
        self.choose_player_pushButton.setObjectName("choose_player_pushButton")
        self.horizontalLayout.addWidget(self.choose_player_pushButton)
        spacerItem10 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem10)
        self.right_player_lineEdit = QtWidgets.QLineEdit(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.right_player_lineEdit.sizePolicy().hasHeightForWidth())
        self.right_player_lineEdit.setSizePolicy(sizePolicy)
        self.right_player_lineEdit.setMinimumSize(QtCore.QSize(100, 40))
        self.right_player_lineEdit.setMaximumSize(QtCore.QSize(100, 40))
        self.right_player_lineEdit.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.right_player_lineEdit.setObjectName("right_player_lineEdit")
        self.horizontalLayout.addWidget(self.right_player_lineEdit)
        self.horizontalLayout_2.addLayout(self.horizontalLayout)
        spacerItem11 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem11)
        self.next_frame_pushButton = QtWidgets.QPushButton(self.frame)
        self.next_frame_pushButton.setMinimumSize(QtCore.QSize(40, 40))
        self.next_frame_pushButton.setMaximumSize(QtCore.QSize(40, 40))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.next_frame_pushButton.setFont(font)
        self.next_frame_pushButton.setObjectName("next_frame_pushButton")
        self.horizontalLayout_2.addWidget(self.next_frame_pushButton)
        spacerItem12 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem12)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_3.addLayout(self.verticalLayout)
        spacerItem13 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.horizontalLayout_3.addItem(spacerItem13)
        self.quit_pushButton = QtWidgets.QPushButton(self.frame)
        self.quit_pushButton.setMinimumSize(QtCore.QSize(60, 60))
        self.quit_pushButton.setMaximumSize(QtCore.QSize(60, 60))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.quit_pushButton.setFont(font)
        self.quit_pushButton.setStyleSheet("border-image:url(:/bg/stop.png);\n"
"background-color: rgba(0,0, 0, 0);")
        self.quit_pushButton.setText("")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/bg/暂停.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.quit_pushButton.setIcon(icon2)
        self.quit_pushButton.setIconSize(QtCore.QSize(50, 50))
        self.quit_pushButton.setObjectName("quit_pushButton")
        self.horizontalLayout_3.addWidget(self.quit_pushButton)
        spacerItem14 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.horizontalLayout_3.addItem(spacerItem14)
        self.horizontalLayout_6.addWidget(self.frame)
        spacerItem15 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_6.addItem(spacerItem15)
        self.verticalLayout_2.addLayout(self.horizontalLayout_6)
        spacerItem16 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout_2.addItem(spacerItem16)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1425, 23))
        self.menubar.setObjectName("menubar")
        self.file_menu = QtWidgets.QMenu(self.menubar)
        self.file_menu.setObjectName("file_menu")
        self.menu = QtWidgets.QMenu(self.file_menu)
        self.menu.setObjectName("menu")
        self.tools_menu = QtWidgets.QMenu(self.menubar)
        self.tools_menu.setObjectName("tools_menu")
        self.help_menu = QtWidgets.QMenu(self.menubar)
        self.help_menu.setObjectName("help_menu")
        MainWindow.setMenuBar(self.menubar)
        self.open_file_action = QtWidgets.QAction(MainWindow)
        self.open_file_action.setObjectName("open_file_action")
        self.open_mask_action = QtWidgets.QAction(MainWindow)
        self.open_mask_action.setObjectName("open_mask_action")
        self.gen_mask_action = QtWidgets.QAction(MainWindow)
        self.gen_mask_action.setObjectName("gen_mask_action")
        self.quit_sys_action = QtWidgets.QAction(MainWindow)
        self.quit_sys_action.setObjectName("quit_sys_action")
        self.functions_action = QtWidgets.QAction(MainWindow)
        self.functions_action.setObjectName("functions_action")
        self.double_view_action = QtWidgets.QAction(MainWindow)
        self.double_view_action.setObjectName("double_view_action")
        self.gen_mask_action_2 = QtWidgets.QAction(MainWindow)
        self.gen_mask_action_2.setObjectName("gen_mask_action_2")
        self.gen_masks_action = QtWidgets.QAction(MainWindow)
        self.gen_masks_action.setObjectName("gen_masks_action")
        self.save_pictures_action = QtWidgets.QAction(MainWindow)
        self.save_pictures_action.setObjectName("save_pictures_action")
        self.save_video_action = QtWidgets.QAction(MainWindow)
        self.save_video_action.setObjectName("save_video_action")
        self.menu.addAction(self.save_pictures_action)
        self.menu.addAction(self.save_video_action)
        self.file_menu.addAction(self.open_file_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.open_mask_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.menu.menuAction())
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.quit_sys_action)
        self.tools_menu.addAction(self.functions_action)
        self.tools_menu.addAction(self.gen_masks_action)
        self.menubar.addAction(self.file_menu.menuAction())
        self.menubar.addAction(self.tools_menu.menuAction())
        self.menubar.addAction(self.help_menu.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "视频智能修复与增强系统"))
        self.previous_frame_pushButton.setText(_translate("MainWindow", "<"))
        self.choose_player_pushButton.setText(_translate("MainWindow", "All"))
        self.next_frame_pushButton.setText(_translate("MainWindow", ">"))
        self.file_menu.setTitle(_translate("MainWindow", "文件"))
        self.menu.setTitle(_translate("MainWindow", "保存"))
        self.tools_menu.setTitle(_translate("MainWindow", "工具"))
        self.help_menu.setTitle(_translate("MainWindow", "帮助"))
        self.open_file_action.setText(_translate("MainWindow", "视频/视频帧"))
        self.open_mask_action.setText(_translate("MainWindow", "mask"))
        self.gen_mask_action.setText(_translate("MainWindow", "生成mask"))
        self.quit_sys_action.setText(_translate("MainWindow", "退出&(Q)"))
        self.functions_action.setText(_translate("MainWindow", "功能"))
        self.double_view_action.setText(_translate("MainWindow", "双视图(默认)"))
        self.gen_mask_action_2.setText(_translate("MainWindow", "mask生成"))
        self.gen_masks_action.setText(_translate("MainWindow", "mask生成"))
        self.save_pictures_action.setText(_translate("MainWindow", "保存为图片"))
        self.save_video_action.setText(_translate("MainWindow", "保存为视频"))


import images_rc
