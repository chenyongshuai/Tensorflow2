# -*- coding: utf-8 -*-
import os
import logging
import traceback
import wave as we
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import scipy.signal as signal
from scipy.fftpack import fft,ifft

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
                logger.info("源文件名：" + path)
                logger.info("WAVE文件名：" + wavepath)
                os.system("D:/Toolsdownload/ffmpeg/ffmpeg-20191219-99f505d-win64-static/bin/ffmpeg -i " + path + " " + wavepath)
        except Exception as e:
            logger.error("输入的path不存在！")
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
                logger.info("源文件名：" + path)
                path = self.transformWave(path)
            else:
                logger.info("WAVE文件名：" + path)
            WAVE = we.open(path)
            sum_sample = WAVE.getparams().nframes
            sample_frequency, audio_sequence = wavfile.read(path)
            logger.info("sum_sample：%s - sample_frequency：%s - audio_sequence：%s" % (sum_sample,sample_frequency, audio_sequence))
            #x_seq = np.arange(0, sum_sample/sample_frequency, 1/sample_frequency)
        except FileNotFoundError as f:
            logger.error("输入的文件路径不存在：" + path)
            traceback.print_exc()
        finally:
            return sum_sample, sample_frequency, audio_sequence
    def showWave(self,x,y):#展示Wav文件图谱
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
                logger.info("x.shape：%s - x.shape：%s" % (x.shape, y.shape))
                raise Exception("输入的两个参数shape不一致！")
        except Exception as e:
            logger.error("x.shape：%s - x.shape：%s" % (x.shape, y.shape))
            traceback.print_exc()
        finally:
            return
    def normalization(self,data):#标准化处理
        """
        :param data: 音频文件的数据内容
        方法： MaxAbs
        :return: 标准化后的数据内容
        """
        try:
            data_R = preprocessing.MaxAbsScaler().fit_transform(data)
        except Exception as e:
            #traceback.print_exc()
            data = data.reshape(-1, 1)
            data_R = preprocessing.MaxAbsScaler().fit_transform(data)
            data_R = data_R.reshape(-1)
        finally:
            logger.info("标准化前的shape：%s 数据：%s " % (data.shape, data))
            logger.info("标准化后的shape：%s 数据：%s " % (data_R.shape, data_R))
            return data_R
    def preEmphasis(self,data):#预加重处理
        """
        :param data：音频文件的数据内容
        方法：Y(n) = X(n) - a * X(n-1)  ### a = 0.9375
        :return: data：预加重后的数据内容
        """
        try:
            logger.info("预加重前的shape：%s 数据：%s " %(data.shape,data))
            a = 0.9375
            data_R = []
            data_R.append(data[0])
            for i in range (1,len(data)):
                r = data[i] - a*data[i-1]
                data_R.append(r)
            data_R = np.asarray(data_R)
            logger.info("预加重后的shape：%s 数据：%s " %(data_R.shape,data_R))
        except AttributeError as a:
            logger.error(a)
            traceback.print_exc()
        return data_R
    def doFraming(self,data,frequency,one_frame_time:float=None,overlap_time:float=None):
        """
        :param data:
        :param frame_length:
        :param overlap_length:
        :return:
        """
        try:
            self.one_frame_time = one_frame_time if one_frame_time is not None else 25.
            self.overlap_time = overlap_time if overlap_time is not None else 10.
            winLength = int(self.one_frame_time / ( 1. / float(frequency) *1000 ))
            overlapNum = int(self.overlap_time / ( 1. / float(frequency) *1000 ))
            logger.info("一帧样本点个数：%s , 帧移样本点数：%s" %(winLength,overlapNum))
            frame_array = []
            numFrame=0
            if len(data) <= winLength:
                numFrame=1
            else:
                numFrame = int(np.ceil((1.0 * len(data) - winLength + overlapNum ) / overlapNum))
            logger.info("总帧数：%s " % (numFrame))
            pad_length = int((numFrame - 1) * overlapNum + winLength)
            logger.info("所有帧加起来总的铺平后的长度：%s " % (pad_length))
            zeros = np.zeros((pad_length - len(data),))
            logger.info("填补长度：%s " % (zeros.shape))
            pad_data = np.concatenate((data, zeros))
            logger.info("填补前的信号长度：%s - 填补后的信号长度：%s " % (data.shape,pad_data.shape))
            indices = np.tile(np.arange(0, winLength), (numFrame, 1)) + np.tile(np.arange(0, numFrame * overlapNum, overlapNum), (winLength, 1)).T
            indices = np.array(indices, dtype=np.int32)
            frames = np.array(pad_data[indices])
            logger.info("分帧后的信号数据为：%s ,%s" % (frames.shape,frames))
        except Exception as e:
            traceback.print_exc()
        finally:
            return winLength,overlapNum,numFrame,frames
    def addWindow(self,frames,winFunc,winLength,numFrame):
        """
        :param frames: 分帧后的音频数据
        :param windowFunc:窗函数
        :return:
        """
        if winFunc == 'hamming':
            func = signal.hamming(winLength)
            #logger.info("func：%s" % func)
            win = np.tile(func, (numFrame, 1))
            #logger.info("win：%s" % win)
            frames = frames*win
            logger.info("加窗-frames：%s ,%s" % (frames.shape,frames))
        return frames
    def doFFT(self,frames):
        """
        :param frames:加窗后的语音信号
        :return:
        """
        print(frames.shape[1])
        frames = abs(fft(frames))/(frames.shape[1]/2)
        logger.info("取模-frames：%s ,%s" % (frames.shape, frames))
        frames = frames[:,0:600]
        logger.info("FFT-frames：%s ,%s" % (frames.shape,frames))
        return frames
















