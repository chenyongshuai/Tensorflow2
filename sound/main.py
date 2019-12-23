# -*- coding: utf-8 -*-
import logging

from sound.SoundHandle import SoundHandle

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(filename)s - %(funcName)s - %(lineno)s - %(levelname)s - %(message)s',
                   datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='log.txt')
logger = logging.getLogger(__name__)
path = "D:/python/workspaces/Tensorflow2.0/sound/enen.wav"
sh = SoundHandle()
sh.readWaveFile(path)
