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
        try:
            wavepath = ""
            if not os.path.exists(path):
                raise FileNotFoundError
            elif os.path.splitext(path)[1] == '.wav':
                raise Exception("需要转换的文件已经是WAVE格式！")
            else:
                wavepath = os.path.splitext(path)[0] + ".wav"
                logging.info("源文件名：" + path)
                logging.info("WAVE文件名：" + wavepath)
                os.system("D:/Toolsdownload/ffmpeg/ffmpeg-20191219-99f505d-win64-static/bin/ffmpeg -i " + path + " " + wavepath)
        except Exception as e:
            logging.error("输入的path不存在！")
            traceback.print_exc()
        finally:
            return wavepath
    def readWaveFile(self,path):#读取WaveFile
        """
        :param path:wavefile的文件路径
        :return: sum_sample：总样本数
                sample_frequency：采样频率
                audio_sequence：数据
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
            logging.info("sample_frequency：%s - audio_sequence：%s" % (sample_frequency, audio_sequence))
            x_seq = np.arange(0, sum_sample/sample_frequency, 1/sample_frequency)
        except FileNotFoundError as f:
            logging.error("输入的文件路径不存在：" + path)
            traceback.print_exc()
        finally:
            return sum_sample, sample_frequency, audio_sequence
    def showWave(self,x,y):
        """
        :param x: 分成多少份
        :param y: 每一份的值
        :return:
        """
        try:
            if x.shape == y.shape:
                plt.plot(x, y, 'blue')
                plt.xlabel("time (s)")
                plt.show()
            else:
                logging.info("x.shape：%s - x.shape：%s" % (x.shape, y.shape))
                raise Exception("输入的两个参数shape不一致！")
        except Exception as e:
            logging.error("x.shape：%s - x.shape：%s" % (x.shape, y.shape))
            traceback.print_exc()
        finally:
            return
    def preEmphasis(self,data):#预加重处理
        """
        :param data：音频文件的数据内容
        方法：Y(n) = X(n) - a * X(n-1)  ### a = 0.9375
        :return: data：预加重后的数据内容
        """
        print()









