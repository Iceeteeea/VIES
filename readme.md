# 视频修复与增强系统

# 1 环境配置

```jsx
pip install -r requirements.txt
```

或者也可以创建一个新的conda环境

```jsx
conda env create -f environment.yaml
conda activate video
```

# 2 预训练模型

通过此[🔗链接](https://1drv.ms/f/s!AiI3TwilOS8Ut1FyPD5HgwKpxfKT?e=xxGZEo)下载所有预训练权重，该链接中，

BasicSR/下的所有权重文件放入本项目的models/sr_models/BasicSR/checkpoints/目录下

DSTT/下的所有权重文件放入本项目的models/inp_models/DSTT/checkpoints/目录下

fastdvdnet/下的所有权重文件放入本项目的models/deno_models/fastdvdnet/checkpoints/目录下

remaster/下的所有权重文件放入本项目的models/restore_models/remaster/checkpoints/目录下

RIFE/下的所有权重文件放入本项目的models/intp_models/RIFE/checkpoints/目录下

SVCNet/下的所有权重文件放入本项目的models/color_models/SVCNet/checkpoints/目录下
GCP/下的所有权重文件放入models/color_models/GCP/checkpoints/目录下

# 3 界面的修改

```
# 主界面: main_win/win_v2/main_win_v7.ui
# 弹窗1: main_win/win_v2/sub1_win_v7.ui   # todo 每次修改完需要在sub1_win_v7.py里加上几句话，详情见py文件
# 弹窗2: main_win/win_v2/sub2_win_v7.ui
# 修改界面时：使用 QtDesigner 修改 .ui 文件
# 调用界面时：先将 .ui 转化为 .py，再调用 .py
# ui 转为 python 命令：python -m PyQt5.uic.pyuic main_win_v1.ui -o main_win_v1.py
```