# -*- coding: utf-8 -*-
import logging
import numpy as np
from sound.SoundHandle import SoundHandle

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(filename)s - %(funcName)s - %(lineno)s - %(levelname)s - %(message)s',
                   datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='log.txt')
logger = logging.getLogger(__name__)
path = "D:/python/workspaces/Tensorflow2.0/sound/enen.wav"
sh = SoundHandle()
#1.读取语音文件数据
sum_sample, sample_frequency, audio_sequence = sh.readWaveFile(path)
#2.正则化语音文件数据
data_R = sh.normalization(audio_sequence)
#3.预加重语音文件数据
data_R = sh.preEmphasis(data_R)
#4.分帧
winLength,overlapNum,numFrame,frames = sh.doFraming(data_R,sample_frequency)
#5.加窗
frames = sh.addWindow(frames,'hamming',winLength,numFrame)
#6.端点检测、过零率
x_seq = np.arange(0, 1200)
sh.showWave(x_seq,frames[1])
#7.FFT快速傅里叶变换、取绝对值或平方值
frames = sh.doFFT(frames)
x_seq = np.arange(0, 600)
sh.showWave(x_seq,frames[1])
#8.Mel滤波

#9.取对数

#10.DCT（Discrete cosine transform）

#11.动态特征（Delta MFCC）

#12.输出特征向量
