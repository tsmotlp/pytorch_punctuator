#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 18:28:30 2019

@author: tsmotlp
"""
import torch
from utils import set_seed, Configs
from run import Trainer, Tester
import warnings
warnings.filterwarnings('always')

if __name__ == '__main__':
    set_seed(seed=123)
    configs = Configs().parse()
    if configs.mode == 'train':
        trainer = Trainer(configs)
        trainer.train()
    else:
        tester = Tester(configs)
        tester.test()
