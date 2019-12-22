# -*- coding: utf-8 -*-
import logging

from sound.SoundHandle import SoundHandle

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s',
                   datefmt='%a, %d %b %Y %H:%M:%S +0000',
                    filename='log.txt')
logger = logging.getLogger(__name__)
path = "D:/python/workspaces/Tensorflow2.0/sound/enen.m4a"
sh = SoundHandle()
sh.transformWave(path)
