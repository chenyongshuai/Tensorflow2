# -*- coding: utf-8 -*-
import os
import logging
logger = logging.getLogger('__name__.SoundHandle')

class SoundHandle:
    def transformWave(self,path):#语音文件格式转换-->wave
        wavepath = os.path.splitext(path)[0]+".wav"
        logging.info("源文件名："+path)
        logging.info("WAVE文件名：" + wavepath)
        os.system("D:/Toolsdownload/ffmpeg/ffmpeg-20191219-99f505d-win64-static/bin/ffmpeg -i " + path + " " + wavepath)




