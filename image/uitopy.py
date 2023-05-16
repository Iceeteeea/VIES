# -*- coding: utf-8 -*-

'''
    【简介】
	qrc ui转换成py的转换工具

'''

import os
import os.path


class QRC_PY():
    def __init__(self):
        # UI文件所在的路径
        self.dir = './'

    # 列出目录下的所有qrc文件
    def listUiFile(self):
        list = []
        files = os.listdir(self.dir)
        for filename in files:
            # print( dir + os.sep + f  )
            # print(filename)
            if os.path.splitext(filename)[1] == '.qrc':
                list.append(filename)

        return list

    # 把后缀为qrc的文件改成后缀为py的文件名
    def transPyFile(self, filename):
        return os.path.splitext(filename)[0] + '_rc.py'

    # 调用系统命令把qrc转换成py
    def runMain(self):
        list = self.listUiFile()
        for rccfile in list:
            pyfile = self.transPyFile(rccfile)
            cmd = 'pyrcc5 -o {pyfile} {rccfile}'.format(pyfile=pyfile, rccfile=rccfile)
            # print(cmd)
            os.system(cmd)


class UI_PY():
    def __init__(self):
        # UI文件所在的路径
        self.dir = './'

    # 列出目录下的所有ui文件
    def listUiFile(self):
        list = []
        files = os.listdir(self.dir)
        for filename in files:
            # print( dir + os.sep + f  )
            # print(filename)
            if os.path.splitext(filename)[1] == '.ui':
                list.append(filename)

        return list

    # 把后缀为ui的文件改成后缀为py的文件名
    def transPyFile(self, filename):
        return os.path.splitext(filename)[0] + '.py'

    # 调用系统命令把ui转换成py
    def runMain(self):
        list = self.listUiFile()
        for uifile in list:
            pyfile = self.transPyFile(uifile)
            cmd = 'pyuic5 -o {pyfile} {uifile}'.format(pyfile=pyfile, uifile=uifile)
            # print(cmd)
            os.system(cmd)


###### 程序的主入口
if __name__ == "__main__":
    Chan1 = QRC_PY()
    Chan1.runMain()
    Chan2 = UI_PY()
    Chan2.runMain()

