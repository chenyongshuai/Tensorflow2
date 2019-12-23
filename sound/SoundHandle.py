# -*- coding: utf-8 -*-
import os
import logging
import traceback
import wave as we
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger('__name__.SoundHandle')
class SoundHandle:
    def transformWave(self,path):#语音文件格式转换-->wave
        """
        :param path:其他文件格式路径
        :return: wavepath  wav文件路径
        """
        wavepath = os.path.splitext(path)[0]+".wav"
        logging.info("源文件名："+path)
        logging.info("WAVE文件名：" + wavepath)
        os.system("D:/Toolsdownload/ffmpeg/ffmpeg-20191219-99f505d-win64-static/bin/ffmpeg -i " + path + " " + wavepath)
        return wavepath
    def readWaveFile(self,path):
        """
        :param path:wavefile的文件路径
        :return:
        """
        try:
            if not os.path.exists(path):
                raise FileNotFoundError
            elif os.path.splitext(path)[1] != '.wav':
                logging.info("源文件名：" + path)
                path = self.transformWave(path)
            else:
                logging.info("WAVE文件名：" + path)
            WAVE = we.open(path)
            sum_sample = WAVE.getparams().nframes
            logging.info("sum_sample：%s" %sum_sample)
            sample_frequency, audio_sequence = wavfile.read(path)
            logging.info("sample_frequency：%s" % sample_frequency)
            logging.info("audio_sequence：%s" % audio_sequence)
            x_seq = np.arange(0, sum_sample/sample_frequency, 1/sample_frequency)
            plt.plot(x_seq, audio_sequence, 'blue')
            plt.xlabel("time (s)")
            plt.show()
        except FileNotFoundError as f:
            logging.info("输入的文件路径不存在：" + path)
            traceback.print_exc()
        finally:
            return sum_sample, sample_frequency, audio_sequence
    def preEmphasis(self,path):#预加重处理
        """
        :param path:输入文件路径
        :return:
        """









