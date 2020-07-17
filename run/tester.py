#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 16:20:07 2019

@author: tsmotlp
"""


from data import get_dataloader
import os
import torch
from sklearn import metrics
from models import LstmPuncModel


class Tester():
    def __init__(self, args):
        super(Tester, self).__init__()
        self.args = args

        # dataloader
        self.dataloader = get_dataloader(self.args)

        # model
        self.device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
        self.model = LstmPuncModel(self.args)


    def test(self):
        with torch.no_grad():
            for idx, data in enumerate(self.dataloader):
                self.model.setup()
                self.model.set_input(data)
                self.model.forward()





