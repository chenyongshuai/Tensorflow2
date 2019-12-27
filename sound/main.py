# -*- coding: utf-8 -*-
import logging
import numpy as np
from sound.SoundHandle import SoundHandle

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(filename)s - %(funcName)s - %(lineno)s - %(levelname)s - %(message)s',
                   datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='log.txt')
logger = logging.getLogger(__name__)
path = "E:/python/Tensorflow2.0/sound/enen.wav"
sh = SoundHandle()
sum_sample, sample_frequency, audio_sequence = sh.readWaveFile(path)
data_R = sh.normalization(audio_sequence)
x_seq = np.arange(0, sum_sample/sample_frequency, 1/sample_frequency)
data_R = sh.preEmphasis(data_R)
sh.doFraming(data_R,sample_frequency)
#sh.showWave(x_seq,data_R)

