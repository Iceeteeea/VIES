# 1: 环境配置
# 参考FuseFormer: https://github.com/ruiliu-ai/FuseFormer
# 安装PyQt5相关包
# 下载 dstt.pth，放入 models/DSTT/checkpoints  下载地址：https://drive.google.com/file/d/1BuSE42QAAUoQAJawbr5mMRXcqRRKeELc/view
# 下载 fuseformer.pth，放入 models/FuseFormer/checkpoints 下载地址：https://drive.google.com/file/d/1Fq3seV2X6dthbjdw4RTNyVd4HH2WlL7g/view

# MiVOS的环境配置参考：https://github.com/hkchengrex/MiVOS
# 预训练模型，本项目中已经下载好了，无需再次下载
# 可以先跑一跑，看看缺啥补啥。


# 2: 修复功能的实现
# 将输入的 视频帧和mask的路径 存入到config/file.json文件中
# 选择模型，点击“开始修复” --> 调用线程 inp_Thread()
# 不要选择 FGT 模型，相关功能还能做好

# 3: 界面的修改
# 主界面: main_win/win_v2/main_win_v7.ui
# 弹窗1: main_win/win_v2/sub1_win_v7.ui   # todo 每次修改完需要在sub1_win_v7.py里加上几句话，详情见py文件
# 弹窗2: main_win/win_v2/sub2_win_v7.ui
# 修改界面时：使用 QtDesigner 修改 .ui 文件
# 调用界面时：先将 .ui 转化为 .py，再调用 .py
# ui 转为 python 命令：python -m PyQt5.uic.pyuic main_win_v1.ui -o main_win_v1.py
