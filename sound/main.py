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
sum_sample, sample_frequency, audio_sequence = sh.readWaveFile(path)
data_R = sh.normalization(audio_sequence)
data_R = sh.preEmphasis(data_R)
winLength,overlapNum,numFrame,frames = sh.doFraming(data_R,sample_frequency)
x_seq = np.arange(0, 1200)
sh.showWave(x_seq,frames[0])
frames = sh.addWindow(frames,'hamming',winLength,numFrame)
sh.showWave(x_seq,frames[0])



