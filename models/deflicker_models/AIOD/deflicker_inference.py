import os
import shutil

class DeflickerInference():
    def __init__(self, input, item_name):
        self.input = input
        self.item_name = item_name

    def main_worker_deflicker(self):
        if not os.path.exists("data/test"):
            os.mkdir("data/test")
        if not os.path.exists('data/test/frames'):
            shutil.copytree(self.input, 'data/test/frames')

        # export_cmd = "export PYTHONPATH=$PWD"
        deflicker_cmd = "python models/deflicker_models/AIOD/test.py --video_frame_folder {} ".format('data/test/frames')
        # os.system(export_cmd)
        os.system(deflicker_cmd)


    def get_item_path(self):
        """
        :param item_name: 项目名称，如pythonProject
        :return:
        """
        # 获取当前所在文件的路径
        cur_path = os.path.abspath(os.path.dirname(__file__))

        # 获取根目录
        return cur_path[:cur_path.find(self.item_name)] + self.item_name

    def inference(self):
        os.environ['PYTHONPATH'] = self.get_item_path()
        self.main_worker_deflicker()

    def environ(self):
        os.environ['PYTHONPATH'] = self.get_item_path()


if __name__ == '__main__':
#     os.environ['PYTHONPATH'] = get_item_path("VIES")
#
#     main_worker_deflicker("data/inpainting/frames")
    DI = DeflickerInference(input="data/上色/breakdance-flare_gray", item_name="VIES")
    DI.inference()